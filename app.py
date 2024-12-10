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
from secrets_manager import get_service_secrets
# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = Flask(__name__)
CORS(app)

secrets = get_service_secrets('gnosis-influencer')

API_KEY = secrets.get('API_KEY')

C_PORT = int(secrets.get('PORT', 5000))
SQLALCHEMY_DATABASE_URI = (
    f"mysql+pymysql://{secrets['MYSQL_USER']}:{secrets['MYSQL_PASSWORD_CONVOS']}"
    f"@{secrets['MYSQL_HOST']}:{secrets['MYSQL_PORT']}/{secrets['MYSQL_DATABASE']}"
)
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

OPENAI_API_KEY = secrets.get('OPENAI_API_KEY')

# Service URLs
PROFILES_API_URL = secrets.get('PROFILES_API_URL')
QUERIES_API_URL = secrets.get('QUERY_API_URL')
GRAPHQL_API_URL = secrets.get('GRAPHQL_API_URL') # http://54.159.168.135:5000/graphql 

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

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
    logging.info(f"Received request data: {data}")
    
    if not data or 'conversation_id' not in data:
        logging.warning("Missing conversation_id in request data.")
        return jsonify({'error': 'conversation_id is required'}), 400

    conversation_id = data['conversation_id']
    content_chunk_id = data.get('content_chunk_id')
    correlation_id = request.headers.get('X-Correlation-ID')  # Get correlation ID from headers
    logging.info(f"Processing conversation_id: {conversation_id}, content_chunk_id: {content_chunk_id}")

    try:
        # Step 1: Fetch the conversation and messages
        conversation = Conversation.query.get(conversation_id)
        if not conversation:
            logging.warning(f"Conversation not found for id: {conversation_id}")
            return jsonify({'error': 'Conversation not found'}), 404

        # Get all messages in the conversation
        messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp).all()
        logging.info(f"Retrieved {len(messages)} messages for conversation_id: {conversation_id}")

        # Step 2: Get AI persona via profiles API
        headers = {'X-API-KEY': API_KEY}
        if correlation_id:
            headers['X-Correlation-ID'] = correlation_id
        ai_profile_resp = requests.get(f"{PROFILES_API_URL}/api/ais/content/{conversation.content_id}", headers=headers)
        if ai_profile_resp.status_code != 200:
            logging.error("Failed to retrieve AI profile")
            logging.error(f"Response: {ai_profile_resp.text}")
            return jsonify({'error': 'Failed to retrieve AI profile'}), 500

        ai_profile = ai_profile_resp.json()
        systems_instructions = ai_profile.get('systems_instructions', '')
        logging.info("Successfully retrieved AI profile and systems instructions.")

        # Step 3: Get the chunk text
        if content_chunk_id:
            # Get chunk text via gnosis-query
            chunk_resp = requests.get(f"{QUERIES_API_URL}/api/chunk/{content_chunk_id}", headers=headers)
            if chunk_resp.status_code != 200:
                logging.error("Failed to retrieve chunk text")
                return jsonify({'error': 'Failed to retrieve chunk text'}), 500
            chunk_data = chunk_resp.json()
            chunk_text = chunk_data['text']
            logging.info(f"Retrieved chunk text for content_chunk_id: {content_chunk_id}")
        else:
            # Use most recent two messages to find similar chunk (concatenate them)
            last_messages = [msg for msg in reversed(messages)][:2]
            if len(last_messages) == 0:
                logging.warning("No user messages found to base AI response on.")
                return jsonify({'error': 'No user message found to base AI response on'}), 400

            # Search for similar chunks using GraphQL
            query = """
            query search($userId: String!, $queryText: String!, $limit: Int!) {
                searchSimilarChunks(userId: $userId, query: $queryText, limit: $limit) {
                    chunkId
                    contentId
                    fileName
                    text
                    similarityScore
                }
            }
            """
            
            variables = {
                "userId": str(conversation.user_id),  # GraphQL expects string
                "queryText": ' '.join([msg.message_text for msg in last_messages]),
                "limit": 1
            }
            
            search_resp = requests.post(
                GRAPHQL_API_URL,
                headers=headers,
                json={
                    "query": query,
                    "variables": variables
                }
            )
            # logging.info(f"GraphQL search response: {search_resp.json()}")
            if search_resp.status_code != 200:
                logging.error("Failed to perform GraphQL search")
                return jsonify({'error': 'Failed to perform search'}), 500

            search_result = search_resp.json()
            if 'errors' in search_result:
                logging.error(f"GraphQL errors: {search_result['errors']}")
                return jsonify({'error': 'Failed to perform search'}), 500

            if not search_result.get('data', {}).get('searchSimilarChunks'):
                logging.warning("No similar content found for AI to respond with.")
                return jsonify({'error': 'No similar content found for AI to respond with'}), 400

            top_result = search_result['data']['searchSimilarChunks'][0]
            content_chunk_id = top_result['chunkId']
            chunk_text = top_result['text']
            logging.info(f"Found similar chunk with id: {content_chunk_id}")

        # Step 4: Prepare conversation context
        conversation_context = []
        user_query = ''
        for msg in messages:
            role = 'user' if msg.sender == SenderType.user else 'assistant'
            conversation_context.append({'role': role, 'content': msg.message_text})
            if msg.sender == SenderType.user:
                user_query = msg.message_text
                logging.info(f"Last user message: {user_query}")

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
        logging.info("Making API call to GPT-4o for response generation.")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt_messages
        )

        response_json = json.loads(response.choices[0].message.content.strip().replace("```json", "").replace("```", ""))
        logging.info("Received response from GPT-4o.")

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
            logging.info(f"AI messages appended to conversation {conversation_id}.")
        except Exception as e:
            logging.error(f"Error while adding AI messages to the database: {str(e)}")
            return jsonify({'error': 'Invalid JSON response from GPT-4o'}), 500                

        # Acknowledgment
        return jsonify({'message': 'AI messages appended to conversation'}), 200

    except Exception as e:
        logging.error(f"Error in post_message_ai: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
    
# add middleware
@app.before_request
def log_request_info():
    logging.info(f"Headers: {request.headers}")
    logging.info(f"Body: {request.get_data()}")

    # for now just check that it has a Authorization header
    if 'X-API-KEY' not in request.headers:
        logging.warning("No X-API-KEY header")
        return jsonify({'error': 'No X-API-KEY'}), 401
    
    x_api_key = request.headers.get('X-API-KEY')
    if x_api_key != API_KEY:
        logging.warning("Invalid X-API-KEY")
        return jsonify({'error': 'Invalid X-API-KEY'}), 401
    else:
        return

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=C_PORT)