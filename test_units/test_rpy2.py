import unittest

class test_rpy2(unittest.TestCase):
    
    def test_import(self):
        try:
            from rpy2.robjects.packages import importr
            success = True
        except:
            success = False
        self.assertTrue(success, "rpy2 was not imported")

    def test_OpVaR(self):
        try:
            from rpy2.robjects.packages import importr
            TGH = importr('OpVaR')
            success = True
        except:
            success = False
        self.assertTrue(success, "OpVaR was not installed correctly")
    
        
if __name__ == '__main__':
    unittest.main()
