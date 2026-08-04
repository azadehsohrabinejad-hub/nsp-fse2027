import sys 
import os 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src')) 
from calculator import add 
 
def test_add(): 
    assert add(2, 3) == 5 
