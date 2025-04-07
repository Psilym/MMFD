

class DatasetCatalog(object):
    dataset_attrs = {
        'placenta_dataset':{
            'dataset_dir': 'data/dataset/placenta_dataset',
            'split_dir': 'data/dataset/placenta_dataset/splits/fewer',
            'ann_file': 'data/dataset/placenta_dataset/splits/fewer/fewer_test30_fewshot100/test_annotations.json',
        },
    }
    print("add all fold catalog.")

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()

