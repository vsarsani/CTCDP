
import unittest
import pandas as pd
import numpy as np
from longimputation import LongImpute

class TestLongImpute(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'projid': [1, 1, 1, 2, 2, 3, 3, 3],
            'fu_year': [0, 1, 2, 0, 1, 0, 1, 2],
            'hba1c': [5.1, np.nan, 5.3, 5.9, np.nan, 6.2, 6.1, np.nan]
        })
        self.imputer = LongImpute(self.df)

    def test_loess_impute(self):
        imputed_df = self.imputer.loess_impute('hba1c')
        self.assertNotEqual(imputed_df['hba1c_imputed_loess'].isna().sum(), self.df['hba1c'].isna().sum())

    def test_rf_impute(self):
        imputed_df = self.imputer.rf_impute('hba1c')
        self.assertNotEqual(imputed_df['hba1c_imputed_rf'].isna().sum(), self.df['hba1c'].isna().sum())

if __name__ == '__main__':
    unittest.main()
