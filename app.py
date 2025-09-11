from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import json
from datetime import datetime, timedelta
from flask import render_template
import random

app = Flask(__name__)
CORS(app)

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', 'your-openrouter-api-key-here')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_openrouter_api(prompt, system_prompt="You are Nipa, a helpful AI assistant for girls. You're friendly, supportive, and knowledgeable about fashion, beauty, wellness, and lifestyle. IMPORTANT: Always use emojis instead of asterisks (*). Never use asterisks for emphasis - use emojis like 💕✨🌟💖 instead. Respond in a girly, encouraging tone with lots of emojis."):
    """Call OpenRouter API with DeepSeek model"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://nipa-app.onrender.com",
            "X-Title": "Nipa - Girly Lifestyle Hub"
        }
        
        data = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print(f"OpenRouter API Error: {response.status_code} - {response.text}")
            return "Sorry babe! I'm having a little technical moment right now. Please try again in a second! 💕✨"
            
    except Exception as e:
        print(f"API Error: {str(e)}")
        return "Sorry babe! I'm having a little technical moment right now. Please try again in a second! 💕✨"


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

# AI Chat Routes
@app.route('/api/chat', methods=['POST'])
def chat():
    """General AI chat endpoint"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        system_prompt = data.get('system_prompt', "You are Nipa, a helpful AI assistant for girls. You're friendly, supportive, and knowledgeable about fashion, beauty, wellness, and lifestyle. IMPORTANT: Always use emojis instead of asterisks (*). Never use asterisks for emphasis - use emojis like 💕✨🌟💖 instead. Respond in a girly, encouraging tone with lots of emojis.")
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        response = call_openrouter_api(message, system_prompt)
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Fashion Routes
@app.route('/api/fashion/outfit', methods=['POST'])
def generate_outfit():
    """Generate outfit recommendations"""
    try:
        data = request.get_json()
        occasion = data.get('occasion', '')
        weather = data.get('weather', '')
        style = data.get('style', '')
        
        prompt = f"""Create a detailed outfit recommendation for:
        - Occasion: {occasion}
        - Weather: {weather}  
        - Style: {style}
        
        Include specific clothing items, colors, accessories, and styling tips. 
        Make it girly and fashionable with lots of emojis! 💕✨"""
        
        system_prompt = "You are Nipa's fashion stylist AI. Create detailed, trendy outfit recommendations with specific items, colors, and styling tips. Always use emojis instead of asterisks. Be encouraging and girly!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'outfit': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Beauty Routes
@app.route('/api/beauty/skincare', methods=['POST'])
def generate_skincare():
    """Generate skincare routine"""
    try:
        data = request.get_json()
        skin_type = data.get('skin_type', '')
        concerns = data.get('concerns', '')
        budget = data.get('budget', '')
        
        prompt = f"""Create a personalized skincare routine for:
        - Skin Type: {skin_type}
        - Main Concerns: {concerns}
        - Budget: {budget}
        
        Include morning and evening routines with specific product recommendations, 
        application order, and tips. Use emojis and be encouraging! 🧴✨"""
        
        system_prompt = "You are Nipa's skincare expert AI. Create detailed skincare routines with specific products and application tips. Always use emojis instead of asterisks. Be knowledgeable and supportive!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'routine': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Wellness Routes
@app.route('/api/wellness/cycle', methods=['POST'])
def calculate_cycle():
    """Calculate menstrual cycle predictions"""
    try:
        data = request.get_json()
        last_period = data.get('last_period', '')
        cycle_length = int(data.get('cycle_length', 28))
        
        # Parse the date
        last_period_date = datetime.strptime(last_period, '%Y-%m-%d')
        
        # Calculate predictions
        next_period = last_period_date + timedelta(days=cycle_length)
        ovulation = last_period_date + timedelta(days=cycle_length - 14)
        fertile_start = ovulation - timedelta(days=5)
        fertile_end = ovulation + timedelta(days=1)
        
        return jsonify({
            'next_period': next_period.isoformat(),
            'ovulation': ovulation.isoformat(),
            'fertile_start': fertile_start.isoformat(),
            'fertile_end': fertile_end.isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wellness/workout', methods=['POST'])
def generate_workout():
    """Generate workout plan"""
    try:
        data = request.get_json()
        goal = data.get('goal', '')
        time = data.get('time', '')
        
        prompt = f"""Create a personalized workout plan for:
        - Fitness Goal: {goal}
        - Available Time: {time} minutes per day
        
        Include specific exercises, sets, reps, and modifications for beginners.
        Make it encouraging and achievable with emojis! 💪✨"""
        
        system_prompt = "You are Nipa's fitness coach AI. Create safe, effective workout plans with clear instructions. Always use emojis instead of asterisks. Be motivating and supportive!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'workout': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Lifestyle Routes
@app.route('/api/lifestyle/mood', methods=['POST'])
def mood_support():
    """Provide mood support and advice"""
    try:
        data = request.get_json()
        mood = data.get('mood', '')
        emoji = data.get('emoji', '')
        
        prompt = f"""The user is feeling {mood} {emoji}. Provide supportive advice, 
        self-care suggestions, and encouraging words. Include activities that might 
        help improve their mood and remind them of their worth. Use lots of emojis! 💕"""
        
        system_prompt = "You are Nipa's emotional support AI. Provide caring, supportive responses with practical self-care advice. Always use emojis instead of asterisks. Be empathetic and uplifting!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/lifestyle/affirmation', methods=['GET'])
def generate_affirmation():
    """Generate daily affirmation"""
    try:
        prompt = """Generate a powerful, personalized daily affirmation for a girl. 
        Make it empowering, confidence-boosting, and beautiful. Include self-love, 
        strength, and positivity. Use emojis! ✨💖"""
        
        system_prompt = "You are Nipa's affirmation generator. Create beautiful, empowering affirmations that boost confidence and self-love. Always use emojis instead of asterisks. Be inspiring and uplifting!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'affirmation': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/lifestyle/selfcare', methods=['POST'])
def generate_selfcare():
    """Generate self-care routine"""
    try:
        data = request.get_json()
        time = data.get('time', '')
        
        prompt = f"""Create a self-care routine for {time} minutes. Include activities 
        for relaxation, pampering, and mental wellness. Make it achievable and 
        luxurious feeling, even on a budget. Use emojis! 🛁💆‍♀️"""
        
        system_prompt = "You are Nipa's self-care expert. Create relaxing, achievable self-care routines that promote wellness and happiness. Always use emojis instead of asterisks. Be nurturing and caring!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'routine': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Shopping Routes
@app.route('/api/shopping/products', methods=['POST'])
def find_products():
    """Find product recommendations"""
    try:
        data = request.get_json()
        category = data.get('category', '')
        budget = data.get('budget', '')
        specific = data.get('specific', '')
        
        prompt = f"""Find the best {category} products within {budget} budget.
        {f'Specifically looking for: {specific}' if specific else ''}
        
        Include product names, price ranges, where to buy, and why they're great.
        Focus on quality and value. Use emojis! 🛍️💎"""
        
        system_prompt = "You are Nipa's shopping assistant. Recommend quality products with good value, including where to find them and why they're worth buying. Always use emojis instead of asterisks. Be helpful and budget-conscious!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'products': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Community Routes
@app.route('/api/community/question', methods=['POST'])
def answer_question():
    """Answer anonymous community questions"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        prompt = f"""Answer this anonymous question with care and wisdom: "{question}"
        
        Provide supportive, non-judgmental advice. Be understanding and helpful.
        Include practical tips and emotional support. Use emojis! 💭💕"""
        
        system_prompt = "You are Nipa's community support AI. Answer questions with empathy, wisdom, and practical advice. Always use emojis instead of asterisks. Be supportive and non-judgmental!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/community/girl-talk', methods=['GET'])
def girl_talk_topic():
    """Generate girl talk discussion topic"""
    try:
        prompt = """Generate a fun, engaging discussion topic for girls to chat about.
        Make it relatable, interesting, and conversation-starting. Could be about 
        relationships, fashion, life experiences, dreams, or fun hypotheticals. 
        Use emojis! 💬✨"""
        
        system_prompt = "You are Nipa's conversation starter generator. Create engaging, fun topics that girls would love to discuss together. Always use emojis instead of asterisks. Be creative and relatable!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'topic': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/community/style-challenge', methods=['GET'])
def style_challenge():
    """Generate weekly style challenge"""
    try:
        prompt = """Create a fun weekly style challenge for girls. Make it creative,
        achievable, and Instagram-worthy. Include specific styling goals and tips
        for completing the challenge. Use emojis! 🏆👗"""
        
        system_prompt = "You are Nipa's style challenge creator. Design fun, creative fashion challenges that are achievable and inspiring. Always use emojis instead of asterisks. Be creative and encouraging!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'challenge': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({'status': 'healthy', 'message': 'Nipa API is running! 💕✨'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)