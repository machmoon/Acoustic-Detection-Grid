# Acoustic Detection System

Classify sounds using YAMNet + Google Gemini API.

## What to Run

```bash
# 1. Create .env file with your API key
cp .env.example .env
# Edit .env and add your Gemini API key

# 2. Run
python yamnet_gemini.py breakin.wav
```

The API key is stored in `.env` (not committed to git).

## Install

```bash
pip install tensorflow tensorflow-hub numpy scipy google-generativeai
```

Or:
```bash
pip install -r requirements.txt
```

## Files

- `yamnet_gemini.py` - **Use this one!** (YAMNet + Gemini AI)
- `yamnet.py` - Basic version (no AI descriptions)

## What You Get

- ✅ Sound classification (521 classes)
- ✅ AI descriptions from Gemini
- ✅ Top 5 predictions with confidence

---

**Just run:** `python yamnet_gemini.py breakin.wav`
