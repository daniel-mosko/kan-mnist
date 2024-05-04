import pandas as pd

if __name__ == "__main__":
    dataset = pd.read_csv(
        "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
    )
    print(dataset)
    