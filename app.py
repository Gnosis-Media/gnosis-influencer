import logging
import requests
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
from enum import Enum
from sqlalchemy.sql.expression import func
from openai import OpenAI
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = Flask(__name__)
CORS(app)

C_PORT = 5012  # Port for gnosis-influencer

# Database configuration (same as gnosis-convos)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://admin:lGrBWwcZJS10NwFBByTK@convos-db.c1ytbjumgtbu.us-east-1.rds.amazonaws.com:3306/conversation_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Initialize OpenAI client
client = OpenAI()

# Models (Copied from gnosis-convos)
class SenderType(Enum):
    user = 'user'
    ai = 'ai'

class Conversation(db.Model):
    __tablename__ = 'conversation'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, nullable=False)
    content_id = db.Column(db.Integer, nullable=False)
    start_date = db.Column(db.DateTime(timezone=True), default=func.now(), nullable=False)
    last_update = db.Column(db.DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade='all, delete-orphan')

class Message(db.Model):
    __tablename__ = 'message'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    sender = db.Column(db.Enum(SenderType), nullable=False)
    content_chunk_id = db.Column(db.Integer, nullable=True)
    message_text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime(timezone=True), default=func.now(), nullable=False)

# Service URLs
PROFILES_API_URL = 'http://localhost:5011'
QUERIES_API_URL = 'http://localhost:5009'

@app.route('/api/message/ai', methods=['POST'])
def post_message_ai():
    """
    Update a conversation with an AI response.

    Expected JSON payload:
    {
        "conversation_id": <int>,
        "content_chunk_id": <int, optional>
    }
    """
    data = request.get_json()
    if not data or 'conversation_id' not in data:
        return jsonify({'error': 'conversation_id is required'}), 400

    conversation_id = data['conversation_id']
    content_chunk_id = data.get('content_chunk_id')

    try:
        # Step 1: Fetch the conversation and messages
        conversation = Conversation.query.get(conversation_id)
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404

        # Get all messages in the conversation
        messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp).all()

        # Step 2: Get AI persona via profiles API
        ai_profile_resp = requests.get(f"{PROFILES_API_URL}/api/ais/content/{conversation.content_id}")
        if ai_profile_resp.status_code != 200:
            logging.error("Failed to retrieve AI profile")
            return jsonify({'error': 'Failed to retrieve AI profile'}), 500

        ai_profile = ai_profile_resp.json()
        systems_instructions = ai_profile.get('systems_instructions', '')

        # Step 3: Get the chunk text
        if content_chunk_id:
            # Get chunk text via gnosis-query
            chunk_resp = requests.get(f"{QUERIES_API_URL}/api/chunk/{content_chunk_id}")
            if chunk_resp.status_code != 200:
                logging.error("Failed to retrieve chunk text")
                return jsonify({'error': 'Failed to retrieve chunk text'}), 500
            chunk_data = chunk_resp.json()
            chunk_text = chunk_data['text']
        else:
            # Use most recent two messages to find similar chunk (concatenate them)
            last_messages = [msg for msg in reversed(messages)][:2]
            if len(last_messages) == 0:
                # No user messages yet; cannot proceed                
                return jsonify({'error': 'No user message found to base AI response on'}), 400

            # Search for similar chunks
            search_resp = requests.get(
                f"{QUERIES_API_URL}/api/search",
                params={
                    'user_id': conversation.user_id,
                    'content_id': conversation.content_id,
                    'query': ' '.join([msg.message_text for msg in last_messages]),
                    'limit': 1
                }
            )
            if search_resp.status_code != 200:
                logging.error("Failed to perform search")
                return jsonify({'error': 'Failed to perform search'}), 500

            search_result = search_resp.json()
            if not search_result['results']:
                return jsonify({'error': 'No similar content found for AI to respond with'}), 400

            top_result = search_result['results'][0]
            content_chunk_id = top_result['chunk_id']
            chunk_text = top_result['text']

        # Step 4: Prepare conversation context
        conversation_context = []
        user_query = ''
        for msg in messages:
            role = 'user' if msg.sender == SenderType.user else 'assistant'
            conversation_context.append({'role': role, 'content': msg.message_text})
            if msg.sender == SenderType.user:
                user_query = msg.message_text
                # Effectively gets the last user message

        # Step 5: Generate AI response
        prompt_messages = [
            {'role': 'system', 'content': systems_instructions},
            *conversation_context
        ]
        # Determine if it's the first message or a reply
        if len(messages) == 0:
            # First message: create social media post
            prompt = f"Write an informative twitter thread that explains the point you're making below."
        else:
            # Subsequent messages: reply to conversation
            prompt = f"Reply to the user's query based on the following content. \nUser query: {user_query}"
        
        prompt += "\nReply in json format as a list of tweets, in the form [{'tweet': 'tweet text'}, {'tweet': 'tweet text'}, ...]"
        prompt += f"\n\nContent: {chunk_text}"

        prompt_messages.append({'role': 'user', 'content': prompt})

        # Make API call to GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt_messages
        )

        response_json = json.loads(response.choices[0].message.content.strip().replace("```json", "").replace("```", ""))
        try:
            for tweet in response_json:                
                # Step 6: Add AI message to the conversation
                ai_message = Message(
                    conversation_id=conversation_id,
                    sender=SenderType.ai,
                    message_text=tweet['tweet'],
                    content_chunk_id=content_chunk_id
                )
                db.session.add(ai_message)
            db.session.commit()
        except:
            logging.error("Invalid JSON response from GPT-4o")
            return jsonify({'error': 'Invalid JSON response from GPT-4o'}), 500                

        # Acknowledgment
        logging.info(f"AI messages appended to conversation {conversation_id}")
        return jsonify({'message': 'AI messages appended to conversation'}), 200

    except Exception as e:
        logging.error(f"Error in post_message_ai: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=C_PORT)