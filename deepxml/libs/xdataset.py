from .dataset_base import DatasetBase, DataPoint


class Dataset(DatasetBase):
    """Dataset to load and use XML-Datasets with OVA classifiers
    """
    def __getitem__(self, index: int) -> DataPoint:
        """Get the data at index"""
        return DataPoint(
            x=self.features[index],
            y=self.labels[index],
            index=index)
 