from typing import Optional, Any

from .dataset_base import DatasetBase, DataPoint


class Dataset(DatasetBase):
    """Dataset to load and use XML-Datasets with sparse 
       classifiers or embeddings
    """
    def __init__(self,
                 data_dir: str,
                 f_features: str,
                 f_labels: str,
                 f_label_features: Optional[str]=None,
                 data: dict={'X': None, 'Y': None, 'Yf': None},
                 mode: str='train',
                 normalize_features: bool=True,
                 normalize_lables: bool=False,
                 feature_type: str='sparse',
                 label_type: str='sparse',
                 max_len: int=-1,
                 *args: Optional[Any],
                 **kwargs: Optional[Any]) -> None:
        super().__init__(data_dir=data_dir,
                         f_features=f_features,
                         data=data,
                         f_label_features=f_label_features,
                         f_labels=f_labels,
                         max_len=max_len,
                         normalize_features=normalize_features,
                         normalize_lables=normalize_lables,
                         feature_type=feature_type,
                         label_type=label_type,
                         mode=mode
                        )        

    def __getitem__(self, index: int) -> DataPoint:
        """Get the data at index"""
        pos_labels, _ = self.labels[index]
        return DataPoint(
            x=self.features[index],
            y=pos_labels,
            yf=None if self.label_features is None else self.label_features[index], 
            index=index)
 