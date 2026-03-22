import pandas as pd

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# -------------------------------
# Handling Missing Values
# -------------------------------
train['sleep_hours'] = train['sleep_hours'].fillna(train['sleep_hours'].median())
train['previous_day_mood'] = train['previous_day_mood'].fillna(train['previous_day_mood'].mode()[0])
train['face_emotion_hint'] = train['face_emotion_hint'].fillna(train['face_emotion_hint'].mode()[0])

# -------------------------------
# Basic Analysis
# -------------------------------
emotion_counts = train['emotional_state'].value_counts()

most_common_emotion = emotion_counts.idxmax()
least_common_emotion = emotion_counts.idxmin()

print("Most common emotion (Topper):", most_common_emotion)
print("Least common emotion:", least_common_emotion)

# -------------------------------
# Prediction Function (Improved)
# -------------------------------
def predict_emotion(row):
    text = str(row['journal_text']).lower()

    # Rule 1: High stress + low sleep
    if row['stress_level'] >= 4 and row['sleep_hours'] <= 5:
        return 'overwhelmed'

    # Rule 2: High energy + positive mood
    elif row['energy_level'] >= 4 and row['previous_day_mood'] == 'happy':
        return 'focused'

    # Rule 3: Low stress + good energy
    elif row['stress_level'] <= 2 and row['energy_level'] >= 3:
        return 'calm'

    # Rule 4: Text-based confusion
    elif 'confused' in text or 'mixed' in text:
        return 'mixed'

    # Rule 5: Face emotion hint
    elif row['face_emotion_hint'] == 'sad':
        return 'restless'

    # Rule 6: Low energy
    elif row['energy_level'] <= 2:
        return 'restless'

    # Default
    else:
        return 'neutral'

# -------------------------------
# Apply Prediction
# -------------------------------
test['predicted_state'] = test.apply(predict_emotion, axis=1)

# -------------------------------
# Output Analysis
# -------------------------------
prediction_counts = test['predicted_state'].value_counts()

print("\nTotal rows predicted:", len(test))
print("\nPrediction distribution:\n", prediction_counts)
print("\nTop predicted emotion:", prediction_counts.idxmax())

# -------------------------------
# Save Output (Important for submission)
# -------------------------------
test[['journal_text', 'predicted_state']].to_csv("submission.csv", index=False)

print("\nSubmission file saved as 'submission.csv'")