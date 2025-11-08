#!/usr/bin/env python3
"""
news_sentiment.py
Standalone scraper + FinBERT sentiment analyzer.

Inputs:
  - ticker (stock code)
  - days (timeline in days)
  - qty  (total per query will be split across 3 queries; final output trimmed to qty)

Behavior:
  - Queries: ["Saham <BASE>", "Info Saham <BASE>", "<BASE> IDX"] where BASE is ticker without suffix (e.g. BBCA)
  - Scrapes Google News headlines (best-effort selector).
  - Deduplicates by title+link.
  - Translates (googletrans) to English when possible.
  - Runs FinBERT (yiyanghkust/finbert-tone) sentiment classification.
  - Prints compact JSON to stdout. Optionally POST to API with --post.
"""
import argparse
import time
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any
import ssl

# SSL FIX for Railway deployment
ssl._create_default_https_context = ssl._create_unverified_context

import requests
from bs4 import BeautifulSoup
from googletrans import Translator
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------- Config ---------
DEFAULT_API_URL = "http://127.0.0.1:8000/news"  # use --post to send
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# --------- helpers ---------
def scrape_google_news(query: str, last_x_days: int, limit: int) -> List[Tuple[str, str]]:
    """
    Scrape headlines+links from news.google.com search page (best-effort).
    Returns list of (title, link).
    """
    q = query.replace(" ", "+")
    url = f"https://news.google.com/search?q={q}+when%3A{last_x_days}d&hl=id&gl=ID&ceid=ID:id"
    headers = {"User-Agent": USER_AGENT}
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[WARN] Failed to fetch Google News for query '{query}': {e}")
        return []
    
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    # flexible selectors because Google News HTML keeps changing
    anchors = soup.select("a.JtKRv, a[href^='./articles'], article a, a.DY5T1d")  # common classes
    for a in anchors:
        title = a.get_text().strip()
        link = a.get("href") or a.get("data-url") or ""
        if not title:
            continue
        if link.startswith("./"):
            link = "https://news.google.com" + link[1:]
        results.append((title, link))
        if len(results) >= limit:
            break
    return results

def dedupe_items(items: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
    seen = set()
    out = []
    for title, link in items:
        key = (title.strip().lower(), (link or "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append((title.strip(), link.strip() if link else ""))
    return out

def load_finbert(device: torch.device):
    try:
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        model.to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"[ERROR] Failed to load FinBERT model: {e}")
        # Return dummy tokenizer/model that will always return neutral
        return None, None

def analyze_finbert(text: str, tokenizer, model, device) -> Tuple[str, Tuple[float,float,float]]:
    # returns (label, (pos, neu, neg)) percentages
    if not text or tokenizer is None or model is None:
        return "Neutral", (0.0, 100.0, 0.0)
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]  # order: positive, neutral, negative
        labs = ["Positive", "Neutral", "Negative"]
        label = labs[int(np.argmax(probs))]
        pct = tuple((probs * 100.0).round(2).tolist())
        return label, pct
    except Exception as e:
        print(f"[ERROR] Failed to analyze sentiment: {e}")
        return "Neutral", (0.0, 100.0, 0.0)

# --------- main flow ---------
def run(ticker: str, days: int, qty: int, post: bool=False, api_url: str=DEFAULT_API_URL, pause: float=0.2) -> Dict[str,Any]:
    # normalize ticker -> base (strip suffix after first dot, e.g. '.JK')
    base = (ticker or "").strip().upper()
    if "." in base:
        base = base.split(".")[0]

    # Prepare queries exactly as you requested
    queries = [f"Saham {base}", f"Info Saham {base}", f"{base} IDX"]

    per_query = max(1, qty)  # per your spec: qty per flavor
    # Collect raw items
    raw = []
    for q in queries:
        try:
            scraped = scrape_google_news(q, days, per_query)
        except Exception as e:
            print(f"[WARN] scrape failed for query '{q}': {e}")
            scraped = []
        raw.extend([(q, t, l) for (t,l) in scraped])
        time.sleep(pause)
    # Dedupe (by title+link)
    dedup_source = [(t,l) for (_,t,l) in raw]
    deduped = dedupe_items(dedup_source)
    # Translate & sentiment
    translator = Translator()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_finbert(device)
    items = []
    for i, (title, link) in enumerate(deduped, start=1):
        # translate (best-effort). If translator fails keep original.
        try:
            translated = translator.translate(title, src='id', dest='en').text
        except Exception:
            translated = title
        try:
            label, (p_pos, p_neu, p_neg) = analyze_finbert(translated, tokenizer, model, device)
        except Exception as e:
            # fallback to neutral
            print(f"[WARN] Sentiment analysis failed for '{title}': {e}")
            label, (p_pos, p_neu, p_neg) = "Neutral", (0.0, 100.0, 0.0)
        items.append({
            "number": i,
            "title": title,
            "link": link,
            "sentiment": label,
            "positive": p_pos,
            "neutral": p_neu,
            "negative": p_neg,
            "translated": translated if translated != title else None,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        time.sleep(pause)
    # Trim to requested qty (user said qty per query but final behavior: limit to qty total)
    final_items = items[:qty]
    result = {
        "ticker": base,
        "timeline_days": int(days),
        "quantity_requested": int(qty),
        "returned_count": len(final_items),
        "items": final_items,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }
    # Optionally POST
    if post:
        try:
            resp = requests.post(api_url, params={"ticker": base}, json=[{
                "title": it["title"],
                "link": it["link"],
                "sentiment": it["sentiment"],
                "positive": it["positive"],
                "neutral": it["neutral"],
                "negative": it["negative"],
                "query": None,
                "timestamp": it["timestamp"]
            } for it in final_items], timeout=30)
            result["post_status"] = resp.status_code
            try:
                result["post_response"] = resp.json()
            except Exception:
                result["post_response"] = resp.text
        except Exception as e:
            result["post_error"] = str(e)
    return result

def main():
    parser = argparse.ArgumentParser(description="Scrape Google News + FinBERT sentiment (standalone).")
    parser.add_argument("ticker", help="Stock code, e.g. BBCA or GOTO.JK")
    parser.add_argument("days", type=int, help="Lookback timeline in days, e.g. 7")
    parser.add_argument("quantity", type=int, help="How many items to return (final trimmed to this)")
    parser.add_argument("--post", action="store_true", help="POST results to API_URL after run")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API endpoint for --post")
    args = parser.parse_args()

    out = run(args.ticker, args.days, args.quantity, post=args.post, api_url=args.api_url)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()