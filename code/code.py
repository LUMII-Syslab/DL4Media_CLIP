#As of now, there's only one huge dirty file.
#Presumably to be divided into a more reasonable structure.

##EasyNMT

#Takes list of english strings.
#Returns list of latvian strings
#model_type:
#  'bad' - use model opus-mt
#  'good' - use model m2m_100_1.2B
easynmt_model_type = None
easynmt_model = None
def translate(texts, model_type, batch_size = 64, progress = False):
  if progress:
    print('Translating.')
    
  global easynmt_model_type
  global easynmt_model
  global easynmt

  if progress:
    print('Getting Model')

  import gc

  from easynmt import EasyNMT

  if easynmt_model_type != model_type:
    if model_type == 'good':
      easynmt_model = EasyNMT('m2m_100_1.2B')
    elif model_type == 'bad':
      easynmt_model = EasyNMT('opus-mt')
    else:
      raise ValueError('EasyNMT model ' + model_type + ' not supported.')
    easynmt_model_type = model_type
    gc.collect(2)
  
  if progress:
    print('Got Model')

  cut_texts = []
  for text in texts:
    cut_texts.append(text[:200])
  results = []
  for offset in range(0, len(texts), batch_size):
    if progress:
      print(offset, '/', len(texts))
    results+= easynmt_model.translate(cut_texts[offset: offset + batch_size], source_lang = 'lv', target_lang = 'en')
  gc.collect(2)

  return results
  
##CLIP

CLIP_configed = False

def config_torch():
    global torch
    import torch
    torch.device('cpu')

#Get clip model
def config_CLIP():
  global context_length
  global CLIP_model
  global CLIP_configed
  global CLIP_preprocess
  global clip
  if not CLIP_configed:
    import clip
    config_torch()
    CLIP_model, CLIP_preprocess = clip.load('ViT-B/32')
    CLIP_configed = True
    context_length = CLIP_model.context_length

#@title

#!pip install ftfy

import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = "bpe_simple_vocab_16e6.txt.gz"):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text


#Takes list of texts
#Returns CLIP text encodings
def clip_encode_text(texts, batch_size = 1024, progress = False):
  config_CLIP()
  tokenizer = SimpleTokenizer()
  text_tokens = [tokenizer.encode(text) for text in texts]
  text_input = torch.zeros(len(text_tokens), CLIP_model.context_length, dtype=torch.long)
  sot_token = tokenizer.encoder['<|startoftext|>']
  eot_token = tokenizer.encoder['<|endoftext|>']

  for i, tokens in enumerate(text_tokens):
      tokens = [sot_token] + tokens[:75] + [eot_token]
      text_input[i, :len(tokens)] = torch.tensor(tokens)

  #text_input = text_input.cuda()

  batch = 1024

  text_features = []

  for offset in range(0, len(texts), batch):
    if progress:
      print(offset)

    with torch.no_grad():
        text_features.append(CLIP_model.encode_text(text_input[offset: offset + batch]).float())

  text_features = torch.cat(text_features).cpu().numpy()

  return text_features

#Takes list of strings
#Returns list of CLIP text encodings
#Option:
#  load - Loads this from google drive
#  rerun - Runs CLIP on strings
#  cache - Runs CLIP on strings and saves it to google drive
#  check - Loads if file exists, caches if not
#  skip - dosen't do anything and returns None
def get_text_encodings(texts, output_name, option = 'load', progress = False):
  if progress:
    print('Getting CLIP text encodings.')
  
  clip_path = ''
    
  import pickle
  pickle_path = clip_path + output_name + 'CLIP_text_encodings.pkl' 
  result = None

  def load():
    nonlocal result
    if not os.path.isfile(pickle_path):
      raise IOError('File ' + pickle_path, ' not found.')
    pkl_file = open(pickle_path, 'rb')
    result = pickle.load(pkl_file)

  def rerun():
    nonlocal result
    result = clip_encode_text(texts, progress = progress)

  def cache():
    rerun()
    
    #Save results
    output_file = open(pickle_path, 'wb')
    pickle.dump(result, output_file)
    output_file.close()

  def check():
    try:
      load()
    except IOError:
      cache()

  if option == 'load':
    load()
  elif option == 'rerun':
    rerun()
  elif option == 'cache':
    cache()
  elif option == 'check':
    check()
  elif option == 'skip':
    return None
  else:
    raise ValueError('Option "' + option, '" doesn\'t exist.')

  return result