import joblib
import pandas as pd

class ConcreteCompressiveStrengthPredictor:
    def __init__(self, bundle_path: str):
        self.bundle = joblib.load(bundle_path)
        self.model = self.bundle["model"]
        self.feature_names = self.bundle["feature_names"]

    def predict(self, cement, blastFurnaceSlag, flyAsh, water,
                superplasticizer, coarseAggregate, fineAggregate, age):
        values = [[
            cement,
            blastFurnaceSlag,
            flyAsh,
            water,
            superplasticizer,
            coarseAggregate,
            fineAggregate,
            age
        ]]
        df = pd.DataFrame(values, columns=self.feature_names)
        return self.model.predict(df)[0]
