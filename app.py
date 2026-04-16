from flask import Flask, render_template, request, jsonify
import json
import os
import threading
import time
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LinearRegression
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

from serial_reader import ArduinoSerialSource

app = Flask(__name__)
DATA_FILE = 'data/water_data.json'

SERIAL_PORT = os.environ.get('ARDUINO_PORT', 'COM4')
SERIAL_BAUD = int(os.environ.get('ARDUINO_BAUD', '9600'))
arduino_source = ArduinoSerialSource(SERIAL_PORT, SERIAL_BAUD)
arduino_source.start()

_last_saved_ts = None
_save_lock = threading.Lock()

def _sensor_save_loop():
    """Every 10 s, flush any new sensor readings from history into water_data.json."""
    global _last_saved_ts
    while True:
        time.sleep(10)
        try:
            snap = arduino_source.snapshot()
            if not snap['connected'] or not snap['history']:
                continue
            with _save_lock:
                all_data = load_data()
                # find points newer than last saved timestamp
                new_points = [
                    p for p in snap['history']
                    if _last_saved_ts is None or p['t'] > _last_saved_ts
                ]
                if not new_points:
                    continue
                prev_level = new_points[0]['v']
                for p in new_points:
                    curr_level = p['v']
                    now = datetime.fromisoformat(p['t'])
                    all_data.append({
                        'date': now.strftime('%Y-%m-%d'),
                        'time': now.strftime('%H:%M'),
                        'tankCapacity': 100.0,
                        'prevLevel': round(prev_level, 2),
                        'currLevel': round(curr_level, 2),
                        'waterUsed': round(max(prev_level - curr_level, 0), 2),
                        'students': 1,
                        'area': 'Sensor',
                        'block': 'Arduino',
                        'source': 'live',
                        'timestamp': p['t']
                    })
                    prev_level = curr_level
                _last_saved_ts = new_points[-1]['t']
                save_data(all_data)
        except Exception:
            pass

threading.Thread(target=_sensor_save_loop, daemon=True, name='SensorSave').start()

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/live-sensor')
def live_sensor():
    return render_template('arduino.html')


@app.route('/api/arduino/live')
def arduino_live():
    return jsonify(arduino_source.snapshot())

@app.route('/api/submit', methods=['POST'])
def submit_data():
    data = request.json
    prev_level = float(data['prevLevel'])
    curr_level = float(data['currLevel'])
    water_used = prev_level - curr_level
    
    entry = {
        'date': data['date'],
        'time': data['time'],
        'tankCapacity': float(data['tankCapacity']),
        'prevLevel': prev_level,
        'currLevel': curr_level,
        'waterUsed': water_used,
        'students': int(data['students']),
        'area': data['area'],
        'block': data['block'],
        'remarks': data.get('remarks', ''),
        'timestamp': datetime.now().isoformat()
    }
    
    all_data = load_data()
    all_data.append(entry)
    save_data(all_data)
    
    return jsonify({'success': True, 'waterUsed': water_used})

@app.route('/api/dates')
def get_dates():
    data = load_data()
    dates  = sorted(set(d['date']  for d in data), reverse=True)
    blocks = sorted(set(d['block'] for d in data))
    return jsonify({'dates': dates, 'blocks': blocks})

@app.route('/api/analytics')
def analytics():
    data = load_data()
    if not data:
        return jsonify({'error': 'No data available'})
    
    # Filter by date / block / zone if provided
    filter_date  = request.args.get('date')
    filter_block = request.args.get('block')
    filter_zone  = request.args.get('zone')
    if filter_date:  data = [d for d in data if d['date']  == filter_date]
    if filter_block: data = [d for d in data if d['block'] == filter_block]
    if filter_zone:  data = [d for d in data if d['area']  == filter_zone]
    if not data:
        return jsonify({'error': 'No data available for the selected filters'})
    
    total_usage = sum(d['waterUsed'] for d in data)
    total_students = sum(d['students'] for d in data)
    avg_per_student = total_usage / total_students if total_students > 0 else 0
    
    # Peak usage time
    time_usage = defaultdict(float)
    for d in data:
        hour = d['time'].split(':')[0]
        time_usage[hour] += d['waterUsed']
    peak_time = max(time_usage.items(), key=lambda x: x[1])[0] if time_usage else '00'
    
    # Usage by area
    area_usage = defaultdict(float)
    for d in data:
        area_usage[d['area']] += d['waterUsed']
    
    # Daily usage
    daily_usage = defaultdict(float)
    for d in data:
        daily_usage[d['date']] += d['waterUsed']
    
    # Time series data
    time_series = [{'time': d['time'], 'usage': d['waterUsed'], 'date': d['date'],
                    'currLevel': d['currLevel'], 'prevLevel': d['prevLevel']} for d in data]

    # Refill detection — currLevel > prevLevel means tank was refilled
    refill_events = []
    for d in data:
        refill_amt = d['currLevel'] - d['prevLevel']
        if refill_amt > 0:
            refill_events.append({
                'date': d['date'], 'time': d['time'],
                'amount': round(refill_amt, 2),
                'label': f"+{round(refill_amt)}L refill"
            })
    
    # Abnormal usage detection
    avg_usage = total_usage / len(data)
    abnormal = any(d['waterUsed'] > avg_usage * 1.4 for d in data)
    abnormal_entries = [{'time': d['time'], 'date': d['date'], 'usage': d['waterUsed']} 
                        for d in data if d['waterUsed'] > avg_usage * 1.4]
    
    # Hourly usage for heatmap
    hourly_usage = defaultdict(float)
    hourly_count = defaultdict(int)
    for d in data:
        hour = int(d['time'].split(':')[0])
        hourly_usage[hour] += d['waterUsed']
        hourly_count[hour] += 1
    
    # Heatmap data
    heatmap_data = []
    for hour in range(24):
        avg_hour_usage = hourly_usage[hour] / hourly_count[hour] if hourly_count[hour] > 0 else 0
        heatmap_data.append({'hour': hour, 'usage': round(avg_hour_usage, 2)})
    
    # Water Efficiency Score
    expected_usage_per_student = 120  # liters per day
    unique_students = len(set(d['students'] for d in data))
    total_days = len(set(d['date'] for d in data))
    expected_total = expected_usage_per_student * (total_students / len(data)) * total_days
    efficiency_score = min(100, (expected_total / total_usage * 100)) if total_usage > 0 else 0
    
    if efficiency_score >= 90:
        efficiency_category = 'Excellent'
    elif efficiency_score >= 70:
        efficiency_category = 'Moderate'
    else:
        efficiency_category = 'Poor'
    
    # Per Student Water Footprint
    per_student_daily = avg_per_student / len(data) if len(data) > 0 else 0
    
    # Tank Refill Prediction
    latest_entry = data[-1] if data else None
    tank_refill_hours = None
    if latest_entry and len(data) > 1:
        recent_data = data[-5:] if len(data) >= 5 else data
        avg_hourly_consumption = sum(d['waterUsed'] for d in recent_data) / len(recent_data)
        remaining_water = latest_entry['currLevel']
        if avg_hourly_consumption > 0:
            tank_refill_hours = round(remaining_water / avg_hourly_consumption, 2)
    
    # Weekly usage
    weekly_usage = sum(daily_usage.values()) if len(daily_usage) <= 7 else sum(list(daily_usage.values())[-7:])

    # Block usage (for bar chart)
    block_usage = defaultdict(float)
    for d in data:
        block_usage[d['block']] += d['waterUsed']

    # Time-of-day buckets for visualization (liters)
    tod_usage = {
        'morning': 0.0,    # 06:00–11:59
        'afternoon': 0.0,  # 12:00–16:59
        'evening': 0.0,    # 17:00–21:59
        'night': 0.0       # else
    }
    for d in data:
        h = int(d['time'].split(':')[0])
        if 6 <= h < 12:
            tod_usage['morning'] += d['waterUsed']
        elif 12 <= h < 17:
            tod_usage['afternoon'] += d['waterUsed']
        elif 17 <= h < 22:
            tod_usage['evening'] += d['waterUsed']
        else:
            tod_usage['night'] += d['waterUsed']

    # Benchmark: per-student vs target (100–120 L/day typical)
    benchmark_target_low = 100
    benchmark_target_high = 120
    
    # Water saved compared to previous day
    water_saved = 0
    if len(daily_usage) >= 2:
        sorted_dates = sorted(daily_usage.keys())
        today_usage = daily_usage[sorted_dates[-1]]
        yesterday_usage = daily_usage[sorted_dates[-2]]
        water_saved = yesterday_usage - today_usage
    
    # Dynamic Conservation suggestions
    suggestions = []
    
    # Time-based analysis
    morning_usage = sum(d['waterUsed'] for d in data if 6 <= int(d['time'].split(':')[0]) <= 9)
    evening_usage = sum(d['waterUsed'] for d in data if 18 <= int(d['time'].split(':')[0]) <= 21)
    night_usage = sum(d['waterUsed'] for d in data if 22 <= int(d['time'].split(':')[0]) or int(d['time'].split(':')[0]) <= 5)
    
    if morning_usage > total_usage * 0.4:
        suggestions.append(f"Morning usage is {round(morning_usage/total_usage*100)}% of total. Stagger bathing schedules between 6-9 AM.")
    
    if evening_usage > total_usage * 0.35:
        suggestions.append(f"Evening peak detected ({round(evening_usage)} L). Distribute activities across different times.")
    
    if night_usage > total_usage * 0.15:
        suggestions.append(f"High night usage detected ({round(night_usage)} L). Check for leaks or unauthorized usage.")
    
    # Area-based analysis
    bathroom_usage = area_usage.get('Bathroom', 0)
    kitchen_usage = area_usage.get('Kitchen', 0)
    laundry_usage = area_usage.get('Laundry', 0)
    cleaning_usage = area_usage.get('Cleaning', 0)
    
    if bathroom_usage > total_usage * 0.5:
        suggestions.append(f"Bathroom usage is {round(bathroom_usage/total_usage*100)}%. Install low-flow shower heads and aerators.")
    
    if kitchen_usage > total_usage * 0.25:
        suggestions.append(f"Kitchen usage is high ({round(kitchen_usage)} L). Use water-efficient dishwashing methods.")
    
    if laundry_usage > total_usage * 0.3:
        suggestions.append(f"Laundry consumes {round(laundry_usage/total_usage*100)}%. Schedule washing hours and use full loads only.")
    
    if cleaning_usage > total_usage * 0.2:
        suggestions.append(f"Cleaning usage is {round(cleaning_usage)} L. Use mops instead of hosing floors.")
    
    # Per-student analysis
    if avg_per_student > 150:
        suggestions.append(f"Average {round(avg_per_student)} L/student is high. Target: 100-120 L/student/day.")
    elif avg_per_student < 80:
        suggestions.append(f"Excellent! Usage at {round(avg_per_student)} L/student is below recommended levels.")
    
    # Abnormal usage
    if abnormal:
        max_usage = max(d['waterUsed'] for d in data)
        suggestions.append(f"⚠️ Abnormal spike detected: {round(max_usage)} L in single entry. Inspect for leaks immediately.")
    
    # Block-specific analysis
    if len(block_usage) > 1:
        max_block = max(block_usage.items(), key=lambda x: x[1])
        min_block = min(block_usage.items(), key=lambda x: x[1])
        if max_block[1] > min_block[1] * 1.5:
            suggestions.append(f"{max_block[0]} uses {round(max_block[1]/min_block[1], 1)}x more than {min_block[0]}. Investigate disparity.")
    
    # Daily trend analysis
    if len(daily_usage) >= 3:
        recent_days = list(daily_usage.values())[-3:]
        if all(recent_days[i] > recent_days[i-1] for i in range(1, len(recent_days))):
            suggestions.append("⚠️ Water usage increasing daily. Review consumption patterns urgently.")
        elif all(recent_days[i] < recent_days[i-1] for i in range(1, len(recent_days))):
            suggestions.append("✓ Great! Water usage decreasing. Keep up conservation efforts.")
    
    # General tips (only if no specific issues found)
    if len(suggestions) < 3:
        suggestions.append("Encourage students to report dripping taps immediately.")
        suggestions.append("Display water conservation posters in common areas.")
        suggestions.append("Conduct weekly water audits to identify wastage.")

    # Curated conservation playbook (shown as actionable steps; order shifts with data)
    conservation_steps = [
        {
            'step': 1,
            'title': 'Fix leaks first',
            'detail': 'A single dripping tap can waste 1,000+ liters per month. Log repairs and re-check tank drop-offs after fixes.',
            'icon': 'fa-tint-slash',
            'priority': 1 if abnormal else 3
        },
        {
            'step': 2,
            'title': 'Shorter, smarter showers',
            'detail': 'Use timer cards or playlists (~5 min). Prefer bucket baths where cultural norms allow; they often use less than long showers.',
            'icon': 'fa-shower',
            'priority': 2 if bathroom_usage > total_usage * 0.45 else 4
        },
        {
            'step': 3,
            'title': 'Kitchen: wash in basins',
            'detail': 'Rinse vegetables and dishes in a filled basin instead of under running water. Only run full dishwasher/racks when available.',
            'icon': 'fa-utensils',
            'priority': 2 if kitchen_usage > total_usage * 0.2 else 5
        },
        {
            'step': 4,
            'title': 'Laundry discipline',
            'detail': 'Full loads only, cold cycles when possible, and one shared laundry window per block to spread peak demand.',
            'icon': 'fa-shirt',
            'priority': 2 if laundry_usage > total_usage * 0.25 else 5
        },
        {
            'step': 5,
            'title': 'Cleaning without the hose',
            'detail': 'Sweep first, mop with minimal buckets, and reuse greywater where safe for non-potable floor cleaning.',
            'icon': 'fa-broom',
            'priority': 2 if cleaning_usage > total_usage * 0.15 else 6
        },
        {
            'step': 6,
            'title': 'Stagger peak times',
            'detail': 'Avoid everyone bathing 6–9 AM / 6–9 PM the same hour. Publish a simple rota so pumps and tanks stay stable.',
            'icon': 'fa-clock',
            'priority': 2 if (morning_usage > total_usage * 0.35 or evening_usage > total_usage * 0.3) else 6
        },
        {
            'step': 7,
            'title': 'Metering awareness',
            'detail': 'Post weekly “liters per student” from this dashboard in notice boards. Friendly competition between blocks often cuts waste.',
            'icon': 'fa-chart-simple',
            'priority': 4 if avg_per_student > benchmark_target_high else 7
        },
        {
            'step': 8,
            'title': 'Rainwater & reuse (where permitted)',
            'detail': 'Capture roof runoff for landscaping/flushing trials. Label pipes clearly and keep potable and non-potable separate.',
            'icon': 'fa-cloud-rain',
            'priority': 8
        }
    ]
    conservation_steps.sort(key=lambda x: (x['priority'], x['step']))

    return jsonify({
        'totalUsage': round(total_usage, 2),
        'avgPerStudent': round(avg_per_student, 2),
        'peakTime': f"{peak_time}:00",
        'areaUsage': dict(area_usage),
        'dailyUsage': dict(daily_usage),
        'timeSeries': time_series,
        'abnormal': abnormal,
        'abnormalEntries': abnormal_entries,
        'suggestions': suggestions,
        'heatmapData': heatmap_data,
        'efficiencyScore': round(efficiency_score, 2),
        'efficiencyCategory': efficiency_category,
        'perStudentDaily': round(per_student_daily, 2),
        'tankRefillHours': tank_refill_hours,
        'weeklyUsage': round(weekly_usage, 2),
        'waterSaved': round(water_saved, 2),
        'blockUsage': {k: round(v, 2) for k, v in block_usage.items()},
        'timeOfDayUsage': {k: round(v, 2) for k, v in tod_usage.items()},
        'benchmarkTargetLow': benchmark_target_low,
        'benchmarkTargetHigh': benchmark_target_high,
        'conservationSteps': conservation_steps,
        'latestTankPct': round(data[-1]['currLevel'] / data[-1]['tankCapacity'] * 100, 1) if data and data[-1].get('tankCapacity') else None,
        'refillEvents': refill_events
    })

@app.route('/api/ai-suggestions', methods=['POST'])
def ai_suggestions():
    groq_key = os.environ.get('GROQ_API_KEY', '')
    if not groq_key:
        return jsonify({'error': 'GROQ_API_KEY not set in .env'}), 500

    d = request.json or {}

    # convert JSON data to natural language for better AI understanding
    leakage_text = 'Yes — immediate inspection required' if d.get('abnormal') else 'No'
    efficiency    = d.get('efficiencyScore', 'N/A')
    category      = d.get('efficiencyCategory', 'N/A')
    per_person    = d.get('avgPerStudent', 'N/A')
    total         = d.get('totalUsage', 'N/A')
    peak          = d.get('peakTime', 'N/A')
    zone          = d.get('topZone', 'N/A')

    natural_language = (
        f"Total water usage is {total} liters. "
        f"There are residents using an average of {per_person} liters per person. "
        f"The peak usage hour is {peak}. "
        f"Leakage detected: {leakage_text}. "
        f"The water efficiency score is {efficiency}% which is rated as {category}. "
        f"The highest water consuming zone is {zone}."
    )

    prompt = (
        f"Analyze the following hostel water usage data:\n\n"
        f"{natural_language}\n\n"
        f"Based on this data, provide exactly 3 short and actionable water conservation suggestions.\n"
        f"Each suggestion must be one clear sentence.\n"
        f"If leakage is detected, one suggestion must be a leakage alert.\n\n"
        f"Respond using this exact JSON format:\n"
        f'{{"suggestions": ['
        f'{{"type": "tip", "text": "suggestion one"}}, '
        f'{{"type": "alert", "text": "suggestion two"}}, '
        f'{{"type": "recommendation", "text": "suggestion three"}}'
        f']}}'
    )

    try:
        client = Groq(api_key=groq_key)
        chat = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[
                {'role': 'system', 'content': 'You are a water conservation expert. You only output valid JSON arrays. Never include markdown, explanations, or any text outside the JSON array.'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.2,
            max_tokens=400,
            response_format={'type': 'json_object'}
        )
        raw = chat.choices[0].message.content.strip()
        print('GROQ RAW RESPONSE:', repr(raw))  # debug

        # strip markdown code fences
        raw = raw.replace('```json', '').replace('```', '').strip()

        # json_object mode may wrap array: {"suggestions": [...]} or {"data": [...]}
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            suggestions = parsed
        elif isinstance(parsed, dict):
            # grab first list value found
            suggestions = next((v for v in parsed.values() if isinstance(v, list)), [])
        else:
            raise ValueError(f'Unexpected response structure: {raw[:200]}')

        # validate each item has required keys
        valid_types = {'tip', 'alert', 'recommendation'}
        cleaned = []
        for s in suggestions:
            if isinstance(s, dict) and 'text' in s:
                cleaned.append({
                    'type': s.get('type', 'tip') if s.get('type') in valid_types else 'tip',
                    'text': str(s['text'])
                })
        if not cleaned:
            raise ValueError('No valid suggestions parsed from response')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    return jsonify({'suggestions': cleaned})


@app.route('/api/predict')
def predict():
    data = load_data()
    if len(data) < 3:
        return jsonify({'error': 'Need at least 3 data points for prediction'})
    
    # Group by date
    daily_usage = defaultdict(float)
    for d in data:
        daily_usage[d['date']] += d['waterUsed']
    
    dates = sorted(daily_usage.keys())
    usage_values = [daily_usage[d] for d in dates]
    
    # Prepare data for linear regression
    X = np.array(range(len(usage_values))).reshape(-1, 1)
    y = np.array(usage_values)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next day
    next_day = len(usage_values)
    prediction = model.predict([[next_day]])[0]
    
    return jsonify({
        'prediction': round(prediction, 2),
        'historicalData': [{'day': i+1, 'usage': usage_values[i]} for i in range(len(usage_values))]
    })

def _start_arduino_serial():
    """Open the serial port once (background thread inside ArduinoSerialSource)."""
    arduino_source.start()


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    _start_arduino_serial()
    # use_reloader=False: one Python process so COM is opened only once (no watcher subprocess).
    app.run(debug=True, use_reloader=False)
