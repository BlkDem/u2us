import json
from pathlib import Path

def load_terms(path):
 data=json.loads(Path(path).read_text(encoding='utf8'))
 return data

def control_hint(text, terms, source='ru', target='en'):
 """Return source text with non-destructive terminology hints for model input."""
 hints=[]
 for source_term, target_value in terms.items():
  values=target_value if isinstance(target_value,list) else [target_value]
  if source_term.lower() in text.lower(): hints.append(f'{source_term} => {" / ".join(values)}')
 return text + (f' <term> {" ; ".join(hints)} </term>' if hints else '')
