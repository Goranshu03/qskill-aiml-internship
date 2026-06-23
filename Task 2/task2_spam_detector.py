"""
============================================================
TASK 2 — Spam Mail Detector
QSkill AI/ML Internship | June 2026
============================================================
Objective : Build a classifier to distinguish spam vs ham
            emails/SMS using NLP feature extraction.
Dataset   : SMS Spam Collection (UCI) — fetched via URL
Models    : Naive Bayes (MultinomialNB) + Logistic Regression
Features  : TF-IDF vectorization
============================================================
"""

import numpy as np
import pandas as pd
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import urllib.request
import os
os.makedirs('outputs', exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

# ── 1. LOAD DATASET ──────────────────────────────────────
print("=" * 60)
print("  TASK 2 — Spam Mail Detector")
print("=" * 60)

# The SMS Spam Collection dataset
URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

print("\n[→] Loading SMS Spam Collection dataset...")
try:
    with urllib.request.urlopen(URL, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(StringIO(raw), sep="\t", header=None, names=["label", "message"])
    print("[✓] Dataset loaded from URL.")
except Exception:
    # Fallback: create a representative synthetic dataset
    print("[!] URL not reachable — using built-in representative dataset.")
    spam_msgs = [
        "WINNER!! You have been selected for a £1000 cash prize. Call now to claim.",
        "Free entry in 2 a weekly competition to win FA Cup final tickets.",
        "Congratulations! You've won a free iPhone. Click here to claim now!",
        "Urgent! Your account will be suspended. Verify now at http://fake-bank.com",
        "You have been awarded a SIM card. Call 08712460324 to receive your award.",
        "CASH PRIZE!! Send your bank details to claim $5000 guaranteed cash reward.",
        "Win a new laptop. Reply with your address to receive your free prize.",
        "Exclusive deal: 90% off on luxury items. Buy now limited stock available.",
        "Your mobile number has been credited with 5000 bonus points. Claim ASAP!",
        "FREE ringtone just text RING to 80082. No subscription required enjoy now.",
        "You are selected for our lucky draw. Reply YES to win $1000 gift voucher.",
        "Final notice: Your loan is approved. Call 0800-FREE to collect £500 cash.",
        "Act now! Buy one get one free on all items. Limited time offer expires soon.",
        "Congrats! U've been selected for a free cruise. Call 09061743810 to confirm.",
        "Alert! Your PayPal account requires verification. Visit secure-paypal-login.net",
        "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 to 87575.",
        "You've been chosen! Claim your free Netflix subscription at fakelink.com",
        "Urgent reply needed. You have unclaimed funds. Contact us immediately!",
        "Hot singles in your area! Click to view profiles and meet tonight.",
        "Your delivery is on hold. Confirm shipping details at parceltrack-fake.com",
        "FREE MESSAGE: Congratulations you have won a 2 week holiday to Benidorm!",
        "PRIVATE! Your 2004 Account Statement for shows 800 un-redeemed pts. Call 08718738034.",
        "Had your mobile 11 months or more? You are entitled to update to the latest colour mobiles with camera for Free!",
        "England v Macedonia – dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077",
        "You are a winner U have been specially selected 2 receive £1000 cash or a 4* holiday (flights inc) speak to a live operator 2 claim 0871",
    ]
    ham_msgs = [
        "Hey, are you coming to the party tonight?",
        "Can you pick up some milk on your way home?",
        "Meeting rescheduled to 3pm tomorrow. See you then.",
        "Happy birthday! Hope you have a wonderful day.",
        "Did you finish the report? Boss is asking.",
        "I'll be late by 20 minutes, stuck in traffic.",
        "Lunch tomorrow sounds great, see you at 1pm.",
        "Can you send me the notes from yesterday's class?",
        "Don't forget about mom's birthday this weekend.",
        "The movie starts at 8. Want to grab dinner before?",
        "Thanks for your help with the project today.",
        "Are you free this weekend for a road trip?",
        "I got the job! Starting next Monday, so excited!",
        "Just wanted to check in, hope you're doing well.",
        "The restaurant was amazing, we should go again.",
        "Call me when you get a chance, need to talk.",
        "Running a bit late, save me a seat please.",
        "Did you see the game last night? Incredible finish!",
        "My flight lands at 9pm. Can you pick me up?",
        "The assignment is due Friday, not Thursday.",
        "Let me know if you need help with the move.",
        "I'm at the library. Where are you?",
        "Mom wants to know if you're coming for Christmas.",
        "The doctor appointment is at 10:30am tomorrow.",
        "Got your message. I'll call you later tonight.",
        "Good morning! How did the interview go?",
        "Sorry I missed your call, was in a meeting.",
        "Want to study together for the exam tonight?",
        "The keys are under the mat, let yourself in.",
        "Just finished cooking, dinner's ready when you are.",
        "Great catching up today! Let's do this again soon.",
        "Hey, can I borrow your notes for tomorrow's class?",
        "No problem, happy to help anytime.",
        "Traffic is terrible today, might be 30 mins late.",
        "Your package was delivered to the front door.",
    ]
    # Balance the dataset (more ham like real world)
    data = (
        [("spam", m) for m in spam_msgs] +
        [("ham",  m) for m in ham_msgs]
    )
    df = pd.DataFrame(data, columns=["label", "message"])

print(f"\nDataset shape : {df.shape}")
print(f"Class distribution:\n{df['label'].value_counts()}\n")

# ── 2. PREPROCESSING ─────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, remove special chars, extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)   # replace URLs
    text = re.sub(r"\d+", " num ", text)               # replace numbers
    text = re.sub(r"[^a-z\s]", " ", text)              # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_msg"] = df["message"].apply(clean_text)
df["label_enc"] = (df["label"] == "spam").astype(int)   # spam=1, ham=0

print("Sample cleaned messages:")
print(df[["label", "clean_msg"]].head(4).to_string(index=False), "\n")

# ── 3. MESSAGE LENGTH ANALYSIS PLOT ──────────────────────
df["msg_length"] = df["message"].apply(len)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Spam Detector — Text Feature Analysis", fontsize=14, fontweight="bold")

# Length distribution
for label, color in [("ham", "#2ecc71"), ("spam", "#e74c3c")]:
    subset = df[df["label"] == label]["msg_length"]
    axes[0].hist(subset, bins=30, alpha=0.6, color=color, label=label)
axes[0].set_title("Message Length Distribution"); axes[0].set_xlabel("Character Count")
axes[0].set_ylabel("Frequency"); axes[0].legend()

# Class balance pie
counts = df["label"].value_counts()
axes[1].pie(counts, labels=counts.index, autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c"], startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Dataset Class Balance")

plt.tight_layout()
plt.savefig("outputs/task2_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] EDA plot saved → task2_eda.png")

# ── 4. TF-IDF VECTORIZATION ──────────────────────────────
X = df["clean_msg"]
y = df["label_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                        stop_words="english", min_df=2)
X_train_v = tfidf.fit_transform(X_train)
X_test_v  = tfidf.transform(X_test)

print(f"Train size : {X_train_v.shape[0]} | Test size : {X_test_v.shape[0]}")
print(f"Vocabulary size (TF-IDF): {X_train_v.shape[1]} features\n")

# ── 5. TRAIN MODELS ──────────────────────────────────────
models = {
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression"    : LogisticRegression(max_iter=300, random_state=42, C=1.0),
}

results = {}
best_name, best_f1, best_preds, best_probs = "", 0, None, None

print("=" * 60)
print("  MODEL RESULTS")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_v, y_train)
    preds = model.predict(X_test_v)
    probs = model.predict_proba(X_test_v)[:, 1]
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test, preds)
    f1   = f1_score(y_test, preds)
    auc  = roc_auc_score(y_test, probs)
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}
    print(f"\n▸ {name}")
    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    if f1 > best_f1:
        best_f1, best_name, best_preds, best_probs = f1, name, preds, probs

print(f"\n🏆 Best model : {best_name} (F1={best_f1:.4f})")
best_model = models[best_name]

print("\n" + "=" * 60)
print(f"  CLASSIFICATION REPORT — {best_name}")
print("=" * 60)
print(classification_report(y_test, best_preds, target_names=["Ham", "Spam"]))

# ── 6. PLOTS: CONFUSION MATRIX + ROC CURVE ───────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Spam Detector — Model Evaluation", fontsize=14, fontweight="bold")

# Confusion matrices
for ax, (name, model) in zip(axes[:2], models.items()):
    preds = model.predict(X_test_v)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], linewidths=0.5)
    ax.set_title(f"{name}\nF1={f1_score(y_test, preds):.4f}", fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

# ROC Curve (both models)
ax = axes[2]
for name, model in models.items():
    probs = model.predict_proba(X_test_v)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve"); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/task2_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[✓] Evaluation plot saved → task2_evaluation.png")

# ── 7. TOP SPAM KEYWORDS ─────────────────────────────────
if isinstance(best_model, MultinomialNB):
    feature_names = tfidf.get_feature_names_out()
    log_prob_diff  = best_model.feature_log_prob_[1] - best_model.feature_log_prob_[0]
    top_idx        = np.argsort(log_prob_diff)[-15:][::-1]
    top_words      = [(feature_names[i], log_prob_diff[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    words  = [w for w, _ in top_words]
    scores = [s for _, s in top_words]
    ax.barh(words[::-1], scores[::-1], color="#e74c3c", edgecolor="white")
    ax.set_title("Top 15 Spam Indicator Words (Naive Bayes log-prob diff)")
    ax.set_xlabel("log P(spam|word) − log P(ham|word)")
    plt.tight_layout()
    plt.savefig("outputs/task2_top_spam_words.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] Top spam words plot saved → task2_top_spam_words.png")

# ── 8. LIVE DEMO PREDICTIONS ─────────────────────────────
print("\n" + "=" * 60)
print("  LIVE DEMO — Predicting New Messages")
print("=" * 60)
demo_msgs = [
    "Congratulations! You have won a free iPhone. Claim now!",
    "Hey, are we still on for lunch tomorrow?",
    "URGENT: Your bank account has been compromised. Click here.",
    "Can you send me the homework assignment please?",
    "Win £1000 cash prize! Text WIN to 80082 to claim your reward.",
]
for msg in demo_msgs:
    vec   = tfidf.transform([clean_text(msg)])
    pred  = best_model.predict(vec)[0]
    prob  = best_model.predict_proba(vec)[0][1]
    label = "🚨 SPAM" if pred == 1 else "✅ HAM "
    print(f"  {label}  ({prob*100:.1f}% spam)  | \"{msg[:60]}\"")

print("\n[✓] Task 2 Complete!\n")
