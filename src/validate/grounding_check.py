import re
from typing import List

def _tokenize(text: str) -> set:
    """
    Advanced tokenizer: Extracts time entities, normalizes them, 
    then processes generic text.
    """
    tokens = set()
    lower_text = text.lower()

    # 1. Extract and Normalize Time Patterns (e.g., 9h00, 9:00, 22h00)
    # Regex looks for Digit + [:h] + 2 Digits
    time_matches = re.findall(r'\d{1,2}[:h]\d{2}', lower_text)
    
    for time_str in time_matches:
        # Normalize 'h' to ':' for consistency
        normalized = time_str.replace('h', ':')
        # Add as a specific token type
        tokens.add(f"time_{normalized}")
        
        # Remove the matched time string from the text to avoid double tokenization
        lower_text = lower_text.replace(time_str, '')

    # 2. Generic Text Tokenization
    # Remove punctuation (keep only alphanumeric/space)
    clean_text = re.sub(r'[^\w\s]', '', lower_text)
    
    # Split by whitespace and filter empty strings
    generic_tokens = set([t for t in clean_text.split() if t])
    
    # 3. Combine
    tokens.update(generic_tokens)
    
    return tokens

def check_grounding(query: str, context_list: List[str], response: str) -> float:
    """
    Calculates a grounding score (Jaccard Index) between the retrieved contexts
    and the generated response.
    
    Jaccard Index = |Intersection(A, B)| / |Union(A, B)|
    """
    if not context_list or not response:
        return 0.0

    # Combine all retrieved contexts into one text blob
    combined_context = " ".join(context_list)
    
    # Tokenize
    context_tokens = _tokenize(combined_context)
    response_tokens = _tokenize(response)
    
    # Calculate Intersection and Union
    intersection = context_tokens.intersection(response_tokens)
    union = context_tokens.union(response_tokens)
    
    if not union:
        return 0.0
        
    jaccard_score = len(intersection) / len(union)
    
    return round(jaccard_score, 4)
