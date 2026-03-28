import structlog
import re
import math
from collections import Counter
from typing import List, Dict

logger = structlog.get_logger()

class TFIDFExtractor:
    @staticmethod
    def _get_sentences(text: str) -> List[str]:
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    @staticmethod
    def _get_words(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    @classmethod
    def extractive_summary(cls, text: str, num_sentences: int = 3) -> str:
        sentences = cls._get_sentences(text)
        if len(sentences) <= num_sentences:
            return text

        # Calculate TF for each sentence
        word_sets = [cls._get_words(s) for s in sentences]
        all_words = [w for ws in word_sets for w in ws]
        word_counts = Counter(all_words)
        
        # Calculate IDF (simple heuristic: rarer words have higher value)
        num_docs = len(sentences)
        idf = {}
        for w in set(all_words):
            doc_freq = sum(1 for ws in word_sets if w in ws)
            idf[w] = math.log10(num_docs / (1 + doc_freq))

        # Score sentences
        scores = []
        for idx, ws in enumerate(word_sets):
            if not ws:
                scores.append((0, idx, sentences[idx]))
                continue
            tf = Counter(ws)
            score = sum((tf[w] / len(ws)) * idf.get(w, 0) for w in ws)
            scores.append((score, idx, sentences[idx]))

        # Get top sentences and sort them back into original order
        top_sentences = sorted(scores, key=lambda x: x[0], reverse=True)[:num_sentences]
        top_sentences.sort(key=lambda x: x[1])
        
        return " ".join(s[2] for s in top_sentences)

    @classmethod
    def extract_keywords(cls, text: str, top_k: int = 5) -> List[str]:
        words = cls._get_words(text)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after', 'beneath', 'under', 'above', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'which', 'who', 'whom', 'whose', 'what', 'where', 'when', 'why', 'how', 'that', 'this', 'these', 'those', 'it', 'its', 'they', 'their', 'them', 'we', 'our', 'us', 'you', 'your', 'he', 'his', 'him', 'she', 'her', 'i', 'my', 'me'}
        filtered = [w for w in words if w not in stopwords and len(w) > 2]
        counts = Counter(filtered)
        return [w for w, c in counts.most_common(top_k)]

class ContextManager:
    def prune_conversation_history(self, messages: List[Dict], max_turns: int = 10) -> List[Dict]:
        """Keep only the last N turns to reduce context bloat."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        
        # Keep last max_turns pairs
        pruned = non_system[-(max_turns * 2):]
        return system_msgs + pruned

    def extract_key_content(self, text: str, max_length: int = 4000) -> str:
        """Extract meaningful content algorithmically."""
        # Use TF-IDF extractive summarizer if text is very long
        if len(text) > max_length:
            target_sentences = max(5, max_length // 100)
            return TFIDFExtractor.extractive_summary(text, num_sentences=target_sentences)
        return text

    def compress_context(self, messages: List[Dict], max_turns: int = 5) -> List[Dict]:
        """Compress conversation history algorithmically."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        
        if len(non_system) <= max_turns * 2:
            return messages
        
        recent = non_system[-(max_turns * 2):]
        older = non_system[:-(max_turns * 2)]
        
        older_summary = self._summarize_turns(older)
        
        # Extract keywords to add as associative tags mapping to older context
        all_older_text = " ".join(m.get("content", "") for m in older)
        keywords = TFIDFExtractor.extract_keywords(all_older_text, top_k=8)
        
        return system_msgs + [{
            "role": "system",
            "content": f"[Earlier conversation summarized: {older_summary}]\n[Keywords: {', '.join(keywords)}]"
        }] + recent

    def _summarize_turns(self, turns: List[Dict]) -> str:
        if not turns:
            return "No earlier context"
        
        # Instead of just taking the first 50 chars, let's use extractive summarization
        full_text = []
        for turn in turns:
            content = turn.get("content", "")
            role = turn.get("role", "")
            full_text.append(f"{role}: {content}")
            
        combined = "\n".join(full_text)
        return TFIDFExtractor.extractive_summary(combined, num_sentences=4)

    def decompress_context(self, messages: List[Dict]) -> List[Dict]:
        """Decompress context by expanding summary markers (RAG hook)."""
        # Currently a placeholder. We will wire this up to the retrieval system later.
        decompressed = []
        for msg in messages:
            content = msg.get("content", "")
            if "[Earlier conversation summarized:" in content:
                # We can inject retrieved context here based on Keywords in the future.
                decompressed.append(msg)
            else:
                decompressed.append(msg)
        return decompressed

    def should_compress(self, messages: List[Dict], token_budget: int = 4000) -> bool:
        """Check if context should be compressed based on estimated token count."""
        text = " ".join(m.get("content", "") for m in messages)
        # Regex to match words and punctuation as distinct tokens
        estimated_tokens = len(re.findall(r"\w+|[^\w\s]", text))
        return estimated_tokens > token_budget
