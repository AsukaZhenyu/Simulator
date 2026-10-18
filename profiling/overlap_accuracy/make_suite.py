"""Write the fixed 18-case grid without reading measurements."""
from pathlib import Path
import contract

if __name__=="__main__":contract.dump(Path(__file__).with_name("suite.json"),contract.suite())
