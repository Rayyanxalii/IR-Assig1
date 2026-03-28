import os, re
import difflib
import nltk
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


class TextCleaner:

    def __init__(self, sw_path=None):
        self.ps = PorterStemmer()
        self.sw = set()
        if sw_path and os.path.exists(sw_path):
            with open(sw_path, encoding='utf-8') as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        self.sw.add(w)

    def scrub(self, txt):
        txt = txt.lower()
        txt = re.sub(r"[^\w\s']", ' ', txt)
        # remove dangling apostrophes
        txt = re.sub(r"'\s|\s'", ' ', txt)
        return txt

    def tokenize(self, txt):
        res = []
        words = self.scrub(txt).split()
        for i, w in enumerate(words):
            w = w.strip("'")
            if not w or w in self.sw:
                continue   # skip stopwords but i still counts (for proximity)
            res.append((self.ps.stem(w), i))
        return res

    def stemQ(self, term):
        # clean up query term before lookup
        t = re.sub(r"[^\w']", '', term.lower().strip())
        if t.endswith('_set'):
            t = t[:-4]   # strip _set notation used in boolean queries
        return self.ps.stem(t)


class SpeechIndex:

    def __init__(self, cleaner):
        self.cleaner = cleaner
        self.inv = {}    # inverted index: term -> set of doc ids
        self.pos = {}    # positional: term -> {docid -> [positions]}
        self.docs = {}   # docid -> filename

    def addDoc(self, did, text, fname):
        self.docs[did] = fname
        for stem, p in self.cleaner.tokenize(text):
            if stem not in self.inv:
                self.inv[stem] = set()
            self.inv[stem].add(did)
            if stem not in self.pos:
                self.pos[stem] = {}
            if did not in self.pos[stem]:
                self.pos[stem][did] = []
            self.pos[stem][did].append(p)

    def _getNum(self, entry):
        m = re.search(r'\d+', entry[1])
        return int(m.group()) if m else 0

    def loadFolder(self, folder):
        if not os.path.isdir(folder):
            print(f"cant find folder: {folder}")
            return

        allFiles = []
        for root, _, fnames in os.walk(folder):
            for fn in fnames:
                if fn.endswith('.txt'):
                    allFiles.append((root, fn))

        allFiles.sort(key=self._getNum)
        print(f"indexing {len(allFiles)} files...")

        for i, (root, fn) in enumerate(allFiles):
            m = re.search(r'\d+', fn)
            did = int(m.group()) if m else i
            try:
                with open(os.path.join(root, fn), 'r', encoding='utf-8', errors='replace') as f:
                    txt = f.read()
            except:
                print(f"  skipping {fn}")
                continue
            self.addDoc(did, txt, fn)
            if (i+1) % 10 == 0:
                print(f"  {i+1}/{len(allFiles)} done")

        print(f"built index with {len(self.inv)} terms")

    def dumpIndexes(self, inv_file='inverted_index.txt', pos_file='positional_index.txt'):
        with open(inv_file, 'w', encoding='utf-8') as f:
            for term in sorted(self.inv.keys()):
                f.write(f"{term} -> {sorted(self.inv[term])}\n")

        with open(pos_file, 'w', encoding='utf-8') as f:
            for term in sorted(self.pos.keys()):
                f.write(f"{term} -> {dict(sorted(self.pos[term].items()))}\n")

        print(f"saved -> {inv_file}")
        print(f"saved -> {pos_file}")

    def loadIndexes(self, inv_file='inverted_index.txt', pos_file='positional_index.txt'):
        import ast
        with open(inv_file, encoding='utf-8') as f:
            for line in f:
                term, _, val = line.strip().partition(' -> ')
                self.inv[term] = set(ast.literal_eval(val))

        with open(pos_file, encoding='utf-8') as f:
            for line in f:
                term, _, val = line.strip().partition(' -> ')
                self.pos[term] = ast.literal_eval(val)

        # rebuild docs map from positional index (any term's doc list works)
        all_ids = set()
        for docset in self.inv.values():
            all_ids.update(docset)
        for did in all_ids:
            if did not in self.docs:
                self.docs[did] = f'speech{did}.txt'

        print(f"loaded {len(self.inv)} terms from {inv_file} and {pos_file}")


class BoolSearch:

    def __init__(self, idx, cleaner):
        self.idx = idx
        self.cleaner = cleaner

    def _all(self):
        return set(self.idx.docs.keys())

    def _stem(self, w):
        s = self.cleaner.stemQ(w)
        if s in self.idx.inv:
            return s
        # try fuzzy match if not found directly
        close = difflib.get_close_matches(s, self.idx.inv.keys(), n=1, cutoff=0.75)
        return close[0] if close else s

    def _getDocs(self, w):
        return self.idx.inv.get(self._stem(w), set())

    def _getPos(self, w, did):
        return self.idx.pos.get(self._stem(w), {}).get(did, [])

    def proxSearch(self, w1, w2, k):
        # positions are raw (stopwords counted), so diff is k+1 not k
        needed = k + 1
        hits = set()
        for did in self._getDocs(w1) & self._getDocs(w2):
            p1 = sorted(self._getPos(w1, did))
            p2 = sorted(self._getPos(w2, did))
            found = False
            for a in p1:
                for b in p2:
                    if b > a and (b - a) == needed:
                        found = True
                        break
                if found:
                    break
            if found:
                hits.add(did)
        return hits

    def _lex(self, q):
        toks = []
        for p in re.findall(r'/\d+|\(|\)|[^\s()/]+', q.strip()):
            up = p.upper()
            if up in ('AND','OR','NOT'):   toks.append(('OP', up))
            elif p == '(':                 toks.append(('LP', p))
            elif p == ')':                 toks.append(('RP', p))
            elif re.match(r'^/\d+$', p):  toks.append(('PROX', int(p[1:])))
            else:                          toks.append(('TERM', p))
        return toks

    def _parseQ(self, q):
        toks = self._lex(q)
        i = [0]

        def cur():
            return toks[i[0]] if i[0] < len(toks) else None

        def eat():
            t = toks[i[0]]; i[0] += 1; return t

        def orExpr():
            left = andExpr()
            while cur() == ('OP','OR'):
                eat(); left = left | andExpr()
            return left

        def andExpr():
            left = notExpr()
            while cur() == ('OP','AND'):
                eat(); left = left & notExpr()
            return left

        def notExpr():
            if cur() == ('OP','NOT'):
                eat()
                return self._all() - atom()
            return atom()

        def atom():
            t = cur()
            if t is None: return set()
            if t[0] == 'LP':
                eat()
                val = orExpr()
                if cur() and cur()[0] == 'RP': eat()
                return val
            if t[0] == 'TERM':
                eat(); return self._getDocs(t[1])
            return set()

        return orExpr()

    def search(self, q):
        q = q.strip()

        # check for  proximity pattern first e.g "word1 word2 /3"
        m = re.match(r'^(.+?)\s+(.+?)\s*/(\d+)$', q)
        if m:
            return sorted(self.proxSearch(m.group(1).strip(), m.group(2).strip(), int(m.group(3))))

        # boolean query
        if any(op in re.split(r'\W+', q.upper()) for op in ('AND','OR','NOT')):
            return sorted(self._parseQ(q))

        parts = q.split()
        if len(parts) == 2:
            # two bare words = adjacent phrase search
            return sorted(self.proxSearch(parts[0], parts[1], 0))

        return sorted(self._getDocs(q))

    def showResults(self, q, res):
        print(f"\n  query : '{q}'")
        print(f"  found : {len(res)} docs")
        print(f"  ids   : {res}")
        print(f"  files : {[self.idx.docs.get(d,'?') for d in res]}")


def runCLI(engine):
    print("\n" + "="*32)
    print("  Boolean IR - Trump Speeches")
    print("  quit to exit")
    print("="*32)
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nexiting"); break
        if not q: continue
        if q.lower() == 'quit': print("bye"); break
        try:
            res = engine.search(q)
            engine.showResults(q, res)
        except Exception as e:
            print(f"  error: {e}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--speeches',  default='speeches')
    ap.add_argument('--stopwords', default='Stopword-List.txt')
    args = ap.parse_args()

    cleaner = TextCleaner(args.stopwords)
    idx = SpeechIndex(cleaner)

    if os.path.exists('inverted_index.txt') and os.path.exists('positional_index.txt'):
        idx.loadIndexes()
    else:
        idx.loadFolder(args.speeches)
        idx.dumpIndexes()

    engine = BoolSearch(idx, cleaner)
    runCLI(engine)