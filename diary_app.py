from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
from markupsafe import Markup
import json
import os
import datetime
import hashlib
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tempfile

# Download NLTK data for sentiment analysis
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'your_secret_key_here'  # Change this to a secure random key

# Add this after creating the Flask app but before defining routes
@app.template_filter('datetime')
def format_datetime(value, format='%Y-%m-%d'):
    """Format a date time to a specified format."""
    if value is None:
        return ""
    return datetime.datetime.strptime(value, '%Y-%m-%d').strftime(format)

@app.template_filter('nl2br')
def nl2br(value):
    """Convert newlines to <br> tags."""
    if value:
        return Markup(value.replace('\n', '<br>'))
    return value

# Helper functions
def get_user_entries(username):
    """Load user entries from file"""
    entries_file = f"entries_{username}.json"
    
    if os.path.exists(entries_file):
        with open(entries_file, "r") as f:
            return json.load(f)
    else:
        return []

def save_user_entries(username, entries):
    """Save user entries to file"""
    entries_file = f"entries_{username}.json"
    
    # Debug information
    print(f"Saving entries to {entries_file}")
    print(f"Number of entries: {len(entries)}")
    print(f"First entry date: {entries[0]['date'] if entries else 'No entries'}")
    
    with open(entries_file, "w") as f:
        json.dump(entries, f)

def analyze_mood(text):
    """Analyze the mood of the text using NLTK's sentiment analyzer"""
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    # Determine mood based on compound score
    if sentiment['compound'] >= 0.05:
        return "Positive"
    elif sentiment['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def calculate_streak(entries):
    """Calculate the current streak"""
    if not entries:
        return 0
    
    # Sort entries by date
    sorted_entries = sorted(entries, key=lambda x: x['date'], reverse=True)
    
    # Get today's date
    today = datetime.datetime.now().date()
    
    # Check if there's an entry for today
    latest_entry_date = datetime.datetime.strptime(sorted_entries[0]['date'], "%Y-%m-%d").date()
    if latest_entry_date != today:
        return 0
    
    # Count consecutive days
    streak = 1
    for i in range(1, len(sorted_entries)):
        prev_date = datetime.datetime.strptime(sorted_entries[i-1]['date'], "%Y-%m-%d").date()
        curr_date = datetime.datetime.strptime(sorted_entries[i]['date'], "%Y-%m-%d").date()
        
        # Check if dates are consecutive
        if (prev_date - curr_date).days == 1:
            streak += 1
        else:
            break
    
    return streak

def calculate_mood(entries):
    """Calculate overall mood based on recent entries"""
    if not entries or len(entries) < 3:
        return "Neutral"
    
    # Get recent entries
    recent_entries = entries[:5] if len(entries) >= 5 else entries
    
    # Count moods
    moods = [entry['mood'] for entry in recent_entries]
    positive_count = moods.count("Positive")
    negative_count = moods.count("Negative")
    neutral_count = moods.count("Neutral")
    
    # Determine overall mood
    if positive_count > negative_count and positive_count > neutral_count:
        return "Positive"
    elif negative_count > positive_count and negative_count > neutral_count:
        return "Negative"
    else:
        return "Neutral"

def generate_mood_chart(entries):
    """Generate mood chart as base64 image showing daily average mood with custom emoji-like markers"""
    if not entries or len(entries) < 3:
        return None
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Group entries by date and calculate average mood
    date_moods = {}
    
    for entry in entries:
        date = entry['date']
        
        # Convert mood to numeric value
        if entry['mood'] == "Positive":
            mood_value = 1
        elif entry['mood'] == "Neutral":
            mood_value = 0
        else:  # Negative
            mood_value = -1
        
        # Add to date_moods dictionary
        if date in date_moods:
            date_moods[date].append(mood_value)
        else:
            date_moods[date] = [mood_value]
    
    # Calculate daily averages
    dates = []
    avg_moods = []
    markers = []
    colors = []
    
    for date, moods in sorted(date_moods.items()):
        date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
        avg_mood = sum(moods) / len(moods)
        
        dates.append(date_obj)
        avg_moods.append(avg_mood)
        
        # Assign marker and color based on mood
        if avg_mood >= 0.5:  # Positive
            markers.append('o')  # Circle for face
            colors.append('#2ecc71')  # Green
        elif avg_mood <= -0.5:  # Negative
            markers.append('o')  # Circle for face
            colors.append('#e74c3c')  # Red
        else:  # Neutral
            markers.append('o')  # Circle for face
            colors.append('#f39c12')  # Orange
    
    # Plot line connecting the points
    ax.plot(dates, avg_moods, linestyle='-', color='#4a6fa5', linewidth=2, alpha=0.7)
    
    # Plot emoji-like markers for each mood
    for i, (date, mood) in enumerate(zip(dates, avg_moods)):
        # Plot the face (circle)
        ax.plot(date, mood, marker=markers[i], markersize=15, color=colors[i], markeredgecolor='black', markeredgewidth=1)
        
        # Add eyes and mouth based on mood
        if mood >= 0.5:  # Positive - happy face
            # Add smile (arc)
            arc_radius = 0.05
            arc_center = (matplotlib.dates.date2num(date), mood - 0.02)
            arc = matplotlib.patches.Arc(arc_center, arc_radius, arc_radius, 
                                        theta1=0, theta2=180, color='black', linewidth=1.5)
            ax.add_patch(arc)
            
            # Add eyes
            left_eye_x = matplotlib.dates.date2num(date) - 0.02
            right_eye_x = matplotlib.dates.date2num(date) + 0.02
            eye_y = mood + 0.02
            ax.plot(left_eye_x, eye_y, 'ko', markersize=3)
            ax.plot(right_eye_x, eye_y, 'ko', markersize=3)
            
        elif mood <= -0.5:  # Negative - sad face
            # Add frown (arc)
            arc_radius = 0.05
            arc_center = (matplotlib.dates.date2num(date), mood + 0.02)
            arc = matplotlib.patches.Arc(arc_center, arc_radius, arc_radius, 
                                        theta1=180, theta2=360, color='black', linewidth=1.5)
            ax.add_patch(arc)
            
            # Add eyes
            left_eye_x = matplotlib.dates.date2num(date) - 0.02
            right_eye_x = matplotlib.dates.date2num(date) + 0.02
            eye_y = mood + 0.02
            ax.plot(left_eye_x, eye_y, 'ko', markersize=3)
            ax.plot(right_eye_x, eye_y, 'ko', markersize=3)
            
        else:  # Neutral - neutral face
            # Add straight mouth
            mouth_left_x = matplotlib.dates.date2num(date) - 0.02
            mouth_right_x = matplotlib.dates.date2num(date) + 0.02
            mouth_y = mood - 0.02
            ax.plot([mouth_left_x, mouth_right_x], [mouth_y, mouth_y], 'k-', linewidth=1.5)
            
            # Add eyes
            left_eye_x = matplotlib.dates.date2num(date) - 0.02
            right_eye_x = matplotlib.dates.date2num(date) + 0.02
            eye_y = mood + 0.02
            ax.plot(left_eye_x, eye_y, 'ko', markersize=3)
            ax.plot(right_eye_x, eye_y, 'ko', markersize=3)
    
    # Add horizontal lines for mood levels
    ax.axhline(y=0, color='#cccccc', linestyle='-', alpha=0.5)
    ax.axhline(y=1, color='#cccccc', linestyle='--', alpha=0.3)
    ax.axhline(y=-1, color='#cccccc', linestyle='--', alpha=0.3)
    
    # Set labels and title
    ax.set_ylabel('Mood')
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_title('Daily Mood Trends')
    
    # Format x-axis dates
    date_locator = matplotlib.dates.AutoDateLocator()
    date_formatter = matplotlib.dates.ConciseDateFormatter(date_locator)
    ax.xaxis.set_major_locator(date_locator)
    ax.xaxis.set_major_formatter(date_formatter)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, markeredgecolor='black', label='Positive'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12', markersize=10, markeredgecolor='black', label='Neutral'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, markeredgecolor='black', label='Negative')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    
    return base64.b64encode(image_png).decode('utf-8')

def generate_word_frequency_chart(entries):
    """Generate word frequency chart as base64 image"""
    if not entries or len(entries) < 3:
        return None
    
    # Get all text from entries
    all_text = " ".join([entry['text'] for entry in entries])
    
    # Remove common words and punctuation
    common_words = {'the', 'and', 'to', 'a', 'of', 'in', 'i', 'is', 'that', 'it', 'for', 'on', 'with', 'as', 'was', 'be', 'this', 'have', 'are', 'not', 'but', 'at', 'from', 'or', 'an', 'my', 'by', 'they', 'you', 'we', 'their', 'his', 'her', 'she', 'he', 'had', 'has', 'been', 'were', 'would', 'could', 'should', 'will', 'can', 'do', 'does', 'did', 'just', 'me', 'them', 'so', 'what', 'who', 'when', 'where', 'why', 'how', 'which', 'there', 'here', 'am', 'if', 'then', 'than', 'your', 'our', 'us', 'very', 'much', 'more', 'most', 'some', 'any', 'all', 'no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'}
    
    # Clean text and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    words = [word for word in words if word not in common_words]
    
    # Count word frequency
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    # Get top 10 words
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if not top_words:
        return None
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot data
    words = [word for word, count in top_words]
    counts = [count for word, count in top_words]
    
    ax.barh(words, counts, color='#4a6fa5')
    
    # Set labels and title
    ax.set_xlabel('Frequency')
    ax.set_title('Most Common Words')
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    
    return base64.b64encode(image_png).decode('utf-8')

def generate_patterns_chart(entries):
    """Generate writing patterns chart as base64 image"""
    if not entries or len(entries) < 5:
        return None
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Get data for chart - word count over time
    dates = []
    word_counts = []
    
    # Sort entries by date
    sorted_entries = sorted(entries, key=lambda x: x['date'])
    
    for entry in sorted_entries:
        date_obj = datetime.datetime.strptime(entry['date'], "%Y-%m-%d")
        dates.append(date_obj)
        word_counts.append(entry['word_count'])
    
    # Plot data
    ax.plot(dates, word_counts, marker='o', linestyle='-', color='#166088')
    
    # Set labels and title
    ax.set_ylabel('Word Count')
    ax.set_title('Writing Patterns Over Time')
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    
    return base64.b64encode(image_png).decode('utf-8')

def generate_pdf_report(username, entries, report_type="all"):
    """Generate a PDF report of diary entries"""
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    temp_filename = temp_file.name
    temp_file.close()
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        temp_filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    normal_style = styles['Normal']
    
    # Create custom styles
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.gray
    )
    
    mood_styles = {
        'Positive': ParagraphStyle(
            'PositiveMood',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.green
        ),
        'Neutral': ParagraphStyle(
            'NeutralMood',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.orange
        ),
        'Negative': ParagraphStyle(
            'NegativeMood',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.red
        )
    }
    
    # Create content elements
    elements = []
    
    # Add title
    report_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if report_type == "monthly":
        title_text = f"Monthly Diary Report - {report_date}"
    elif report_type == "mood":
        title_text = f"Mood Analysis Report - {report_date}"
    else:
        title_text = f"Complete Diary Report - {report_date}"
    
    elements.append(Paragraph(title_text, title_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add user info
    elements.append(Paragraph(f"User: {username}", normal_style))
    elements.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Paragraph(f"Total Entries: {len(entries)}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add mood chart if available and report type is appropriate
    if report_type in ["all", "mood"] and len(entries) >= 3:
        try:
            mood_chart_base64 = generate_mood_chart(entries)
            if mood_chart_base64:
                # Convert base64 to image data
                img_data = base64.b64decode(mood_chart_base64)
                
                # Create a BytesIO object instead of a temporary file
                img_io = BytesIO(img_data)
                
                # Add image to PDF directly from BytesIO
                elements.append(Paragraph("Mood Trends", heading_style))
                elements.append(Spacer(1, 0.1*inch))
                img = Image(img_io, width=6*inch, height=3*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.25*inch))
        except Exception as e:
            # If there's an error with the chart, just skip it
            print(f"Error generating mood chart: {e}")
            elements.append(Paragraph("Mood chart could not be generated", normal_style))
            elements.append(Spacer(1, 0.25*inch))
    
    # Sort entries by date (newest first)
    sorted_entries = sorted(entries, key=lambda x: x['date'], reverse=True)
    
    # Filter entries based on report type
    if report_type == "monthly":
        # Get entries from the current month
        current_month = datetime.datetime.now().month
        current_year = datetime.datetime.now().year
        sorted_entries = [
            entry for entry in sorted_entries 
            if datetime.datetime.strptime(entry['date'], '%Y-%m-%d').month == current_month
            and datetime.datetime.strptime(entry['date'], '%Y-%m-%d').year == current_year
        ]
    elif report_type == "mood":
        # Group entries by mood
        elements.append(Paragraph("Entries by Mood", heading_style))
        elements.append(Spacer(1, 0.1*inch))
        
        for mood in ["Positive", "Neutral", "Negative"]:
            mood_entries = [entry for entry in sorted_entries if entry['mood'] == mood]
            if mood_entries:
                elements.append(Paragraph(f"{mood} Entries ({len(mood_entries)})", styles['Heading2']))
                elements.append(Spacer(1, 0.1*inch))
                
                for entry in mood_entries:
                    date_obj = datetime.datetime.strptime(entry['date'], '%Y-%m-%d')
                    formatted_date = date_obj.strftime("%A, %B %d, %Y")
                    
                    elements.append(Paragraph(formatted_date, date_style))
                    elements.append(Paragraph(entry['text'], normal_style))
                    elements.append(Spacer(1, 0.2*inch))
                
                elements.append(Spacer(1, 0.1*inch))
        
        # Build the PDF
        doc.build(elements)
        return temp_filename
    
    # Add entries section for all and monthly reports
    if report_type in ["all", "monthly"]:
        elements.append(Paragraph("Diary Entries", heading_style))
        elements.append(Spacer(1, 0.1*inch))
        
        for entry in sorted_entries:
            date_obj = datetime.datetime.strptime(entry['date'], '%Y-%m-%d')
            formatted_date = date_obj.strftime("%A, %B %d, %Y")
            
            # Create a table for each entry
            data = [
                [Paragraph(formatted_date, date_style), 
                 Paragraph(f"Mood: {entry['mood']}", mood_styles[entry['mood']])],
                [Paragraph(entry['text'], normal_style), ""]
            ]
            
            t = Table(data, colWidths=[5*inch, 1*inch])
            t.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('SPAN', (0, 1), (1, 1)),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('TOPPADDING', (0, 1), (-1, 1), 5),
                ('BOTTOMPADDING', (0, 1), (-1, 1), 15),
                ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.lightgrey),
            ]))
            
            elements.append(t)
            elements.append(Spacer(1, 0.1*inch))
    
    # Build the PDF
    try:
        doc.build(elements)
        return temp_filename
    except Exception as e:
        print(f"Error building PDF: {e}")
        flash("Error generating PDF report", "error")
        return None

# Routes
@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if not username or not password:
            flash('Please enter both username and password', 'error')
            return redirect(url_for('login'))
        
        # Hash the password for security
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Check if user exists
        users_file = "users.json"
        if os.path.exists(users_file):
            with open(users_file, "r") as f:
                users = json.load(f)
                
            if username in users and users[username]["password"] == hashed_password:
                session['username'] = username
                session['name'] = users[username]["name"]
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password', 'error')
        else:
            flash('No registered users found', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form['fullname']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validate inputs
        if not fullname or not username or not password or not confirm_password:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        # Hash the password for security
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Check if username already exists
        users_file = "users.json"
        users = {}
        
        if os.path.exists(users_file):
            with open(users_file, "r") as f:
                users = json.load(f)
        
        if username in users:
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        # Add new user
        users[username] = {
            "name": fullname,
            "password": hashed_password
        }
        
        # Save to file
        with open(users_file, "w") as f:
            json.dump(users, f)
        
        flash('Registration successful! You can now login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('name', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    # Calculate stats
    entry_count = len(entries)
    streak = calculate_streak(entries)
    
    # Calculate average word count
    avg_words = 0
    if entries:
        avg_words = sum(entry['word_count'] for entry in entries) // len(entries)
    
    # Get overall mood
    overall_mood = calculate_mood(entries)
    
    # Get recent entries (up to 5)
    recent_entries = sorted(entries, key=lambda x: x['date'], reverse=True)[:5]
    
    # Generate mood chart
    mood_chart = generate_mood_chart(entries)
    
    return render_template('dashboard.html', 
                          name=session['name'],
                          entry_count=entry_count,
                          streak=streak,
                          avg_words=avg_words,
                          overall_mood=overall_mood,
                          recent_entries=recent_entries,
                          mood_chart=mood_chart)

@app.route('/diary', methods=['GET', 'POST'])
def diary():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        text = request.form['diary_text'].strip()
        entry_date = request.form['entry_date'].strip()
        
        if not text:
            flash('Please enter some text before saving', 'error')
            return redirect(url_for('diary'))
        
        # Validate date format
        try:
            datetime.datetime.strptime(entry_date, '%Y-%m-%d')
        except ValueError:
            flash('Invalid date format. Please use YYYY-MM-DD format.', 'error')
            return redirect(url_for('diary'))
        
        # Create entry object
        entry = {
            "id": int(datetime.datetime.now().timestamp()),
            "date": entry_date,
            "text": text,
            "word_count": len(text.split()),
            "mood": analyze_mood(text)
        }
        
        # Load existing entries
        entries = get_user_entries(session['username'])
        
        # Add new entry
        entries.append(entry)
        
        # Save entries
        save_user_entries(session['username'], entries)
        
        flash('Entry saved successfully!', 'success')
        return redirect(url_for('diary'))
    
    # Get today's date in YYYY-MM-DD format for the date input
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Format date for display
    display_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
    
    return render_template('diary.html', today_date=today_date, date=display_date)

@app.route('/insights')
def insights():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    # Generate charts
    mood_chart = generate_mood_chart(entries)
    word_chart = generate_word_frequency_chart(entries)
    patterns_chart = generate_patterns_chart(entries)
    
    return render_template('insights.html', 
                          mood_chart=mood_chart,
                          word_chart=word_chart,
                          patterns_chart=patterns_chart)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'update_profile':
            fullname = request.form['fullname']
            
            if not fullname:
                flash('Please enter your full name', 'error')
                return redirect(url_for('settings'))
            
            # Update user profile
            users_file = "users.json"
            with open(users_file, "r") as f:
                users = json.load(f)
            
            users[session['username']]['name'] = fullname
            
            with open(users_file, "w") as f:
                json.dump(users, f)
            
            # Update session
            session['name'] = fullname
            
            flash('Profile updated successfully!', 'success')
            
        elif action == 'change_password':
            current_password = request.form['current_password']
            new_password = request.form['new_password']
            confirm_password = request.form['confirm_password']
            
            if not current_password or not new_password or not confirm_password:
                flash('Please fill in all password fields', 'error')
                return redirect(url_for('settings'))
            
            if new_password != confirm_password:
                flash('New passwords do not match', 'error')
                return redirect(url_for('settings'))
            
            # Hash passwords
            hashed_current = hashlib.sha256(current_password.encode()).hexdigest()
            hashed_new = hashlib.sha256(new_password.encode()).hexdigest()
            
            # Check current password
            users_file = "users.json"
            with open(users_file, "r") as f:
                users = json.load(f)
            
            if users[session['username']]['password'] != hashed_current:
                flash('Current password is incorrect', 'error')
                return redirect(url_for('settings'))
            
            # Update password
            users[session['username']]['password'] = hashed_new
            
            with open(users_file, "w") as f:
                json.dump(users, f)
            
            flash('Password changed successfully!', 'success')
            
        elif action == 'export_data':
            # Load user entries
            entries = get_user_entries(session['username'])
            
            if not entries:
                flash('No entries to export', 'error')
                return redirect(url_for('settings'))
            
            # Create export data
            export_data = {
                "user": session['name'],
                "exported_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "entries": entries
            }
            
            # Save to file
            export_file = f"export_{session['username']}_{datetime.datetime.now().strftime('%Y%m%d')}.json"
            
            with open(export_file, "w") as f:
                json.dump(export_data, f, indent=2)
            
            flash(f'Data exported successfully to {export_file}', 'success')
    
    # Get user data
    users_file = "users.json"
    with open(users_file, "r") as f:
        users = json.load(f)
    
    fullname = users[session['username']]['name']
    
    return render_template('settings.html', fullname=fullname, username=session['username'])

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    # This would normally use a speech recognition API
    # For now, we'll just return a placeholder response
    return jsonify({"text": "This is a placeholder for speech recognition."})

@app.route('/entries')
def entries():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    # Sort entries by date (newest first)
    entries = sorted(entries, key=lambda x: x['date'], reverse=True)
    
    return render_template('entries.html', entries=entries)

@app.route('/entry/<int:entry_id>')
def view_entry(entry_id):
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    # Find the specific entry
    entry = next((e for e in entries if e['id'] == entry_id), None)
    
    if not entry:
        flash('Entry not found', 'error')
        return redirect(url_for('view_entries'))
    
    return render_template('view_entry.html', entry=entry)

@app.route('/entry/edit/<int:entry_id>', methods=['GET', 'POST'])
def edit_entry(entry_id):
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    # Find the specific entry
    entry_index = next((i for i, e in enumerate(entries) if e['id'] == entry_id), None)
    
    if entry_index is None:
        flash('Entry not found', 'error')
        return redirect(url_for('view_entries'))
    
    if request.method == 'POST':
        text = request.form['diary_text'].strip()
        new_date = request.form['entry_date']
        
        # Debug information
        print(f"Form data - Text: {text}, Date: {new_date}")
        print(f"Original entry date: {entries[entry_index]['date']}")
        
        if not text:
            flash('Please enter some text before saving', 'error')
            return redirect(url_for('edit_entry', entry_id=entry_id))
        
        # Validate date format
        try:
            datetime.datetime.strptime(new_date, '%Y-%m-%d')
        except ValueError:
            flash('Invalid date format. Please use YYYY-MM-DD format.', 'error')
            return redirect(url_for('edit_entry', entry_id=entry_id))
        
        # Update entry
        entries[entry_index]['text'] = text
        entries[entry_index]['date'] = new_date
        entries[entry_index]['word_count'] = len(text.split())
        entries[entry_index]['mood'] = analyze_mood(text)
        
        # Debug information after update
        print(f"Updated entry date: {entries[entry_index]['date']}")
        
        # Save entries
        save_user_entries(session['username'], entries)
        
        flash('Entry updated successfully!', 'success')
        return redirect(url_for('view_entry', entry_id=entry_id))
    
    return render_template('edit_entry.html', entry=entries[entry_index])

@app.route('/entry/delete/<int:entry_id>', methods=['POST'])
def delete_entry(entry_id):
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    # Find and remove the specific entry
    entries = [e for e in entries if e['id'] != entry_id]
    
    # Save entries
    save_user_entries(session['username'], entries)
    
    flash('Entry deleted successfully!', 'success')
    return redirect(url_for('entries'))

@app.route('/entries/delete_multiple', methods=['POST'])
def delete_multiple_entries():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Get entry IDs to delete
    entry_ids = request.form.getlist('entry_ids[]')
    
    if not entry_ids:
        flash('No entries selected for deletion', 'error')
        return redirect(url_for('entries'))
    
    # Convert IDs to integers
    entry_ids = [int(id) for id in entry_ids]
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    # Count entries before deletion
    entries_count_before = len(entries)
    
    # Remove selected entries
    entries = [e for e in entries if e['id'] not in entry_ids]
    
    # Count deleted entries
    deleted_count = entries_count_before - len(entries)
    
    # Save updated entries
    save_user_entries(session['username'], entries)
    
    flash(f'{deleted_count} entries deleted successfully!', 'success')
    return redirect(url_for('entries'))

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    report_type = request.form.get('report_type', 'all')
    
    # Load user entries
    entries = get_user_entries(session['username'])
    
    if not entries:
        flash('No entries to generate report', 'error')
        return redirect(url_for('settings'))
    
    try:
        # Generate PDF report
        pdf_path = generate_pdf_report(session['name'], entries, report_type)
        
        if not pdf_path:
            flash('Failed to generate report', 'error')
            return redirect(url_for('settings'))
        
        # Generate a filename for the download
        if report_type == "monthly":
            filename = f"monthly_diary_report_{datetime.datetime.now().strftime('%Y_%m')}.pdf"
        elif report_type == "mood":
            filename = f"mood_analysis_report_{datetime.datetime.now().strftime('%Y_%m_%d')}.pdf"
        else:
            filename = f"diary_report_{datetime.datetime.now().strftime('%Y_%m_%d')}.pdf"
        
        # Send the file
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
    except Exception as e:
        print(f"Error in generate_report: {e}")
        flash('An error occurred while generating the report', 'error')
        return redirect(url_for('settings'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)








































