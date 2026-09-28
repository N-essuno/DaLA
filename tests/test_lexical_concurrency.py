from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time
import unittest
from dala.lexical_resources import SerializedDictionary

class LexicalConcurrencyTests(unittest.TestCase):
    def test_native_handle_has_one_caller_and_preserves_results(self):
        guard=Lock();active=0;maximum=0
        def native(word):
            nonlocal active,maximum
            with guard:active+=1;maximum=max(maximum,active)
            time.sleep(.001)
            with guard:active-=1
            return word=='valid'
        dictionary=SerializedDictionary(native)
        words=['valid','invalid']*32
        with ThreadPoolExecutor(max_workers=8) as pool:values=list(pool.map(dictionary.lookup,words))
        self.assertEqual(values,[word=='valid' for word in words]);self.assertEqual(maximum,1)

if __name__=='__main__':unittest.main()
