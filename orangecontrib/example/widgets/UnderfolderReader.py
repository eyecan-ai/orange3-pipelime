from os import replace
from pathlib import Path
from AnyQt.QtWidgets import QLabel
from Orange.data.variable import ContinuousVariable, DiscreteVariable
from Orange.widgets.widget import OWWidget, Output, Input
import Orange
from pipelime.filesystem.toolkit import FSToolkit
from pipelime.sequences.samples import FileSystemSample, Sample
import torch
from torchvision.models.resnet import resnet18
from pipelime.sequences.readers.filesystem import UnderfolderReader
from Orange.data import Table, Domain, StringVariable
import numpy as np
import pandas as pd
from choixe.configurations import XConfig


class EyecanUnderfolderReader(OWWidget):
    DATA_REMAP = {
        int: ContinuousVariable,
        float: ContinuousVariable,
        str: StringVariable,
        bool: DiscreteVariable,
    }
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "Eyecan Underfolder Reader"
    icon = "icons/lime.svg"
    want_main_area = False

    class Outputs:
        output = Output("output", Table)

    def __init__(self):
        super().__init__()

        self._reader = UnderfolderReader(folder="/tmp/crops")
        self._images = None
        label = QLabel(f"Dataset size: {len(self._reader)}")
        self.controlArea.layout().addWidget(label)

        metas_data = []
        attributes_data = []
        attributes, metas = None, None
        for sample_idx, sample in enumerate(self._reader):
            # if sample_idx > 100:
            #     break
            sample: FileSystemSample

            # Process sample and extract meta and attributes, along with a purged addressable sample dict
            processed = self._process_sample(sample)
            attributes = processed["attributes"]
            metas = processed["metas"]
            purged_sample = processed["purged_sample"]

            # Fill attributes
            attributes_row = []
            for attribute_idx, attribute in enumerate(attributes):
                print("GETTING", attribute, "FROM", list(purged_sample.keys()))
                attributes_row.append(purged_sample[attribute.name])
            attributes_data.append(attributes_row)

            # Fille meta
            meta_row = []
            for meta_idx, meta_attribute in enumerate(metas):
                meta_row.append(purged_sample[meta_attribute.name])
            metas_data.append(meta_row)

        # Reshape data as table/s
        metas_data = np.array(metas_data).reshape(-1, len(metas))
        attributes_data = np.array(attributes_data).reshape(-1, len(attributes))

        # Create domain
        domain = Domain(attributes, metas=metas)
        data = Table.from_numpy(X=attributes_data, domain=domain, metas=metas_data)

        self.Outputs.output.send(data)

    def _process_sample(self, sample: FileSystemSample) -> dict:

        sample_copy = sample.copy()
        keys = list(sample_copy.keys())

        purged_sample = {}
        variables = []
        for key in keys:
            filename = Path(sample.filesmap[key])
            if FSToolkit.is_image_file(filename):
                del sample_copy[key]

                image_variable = StringVariable.make(key)
                image_variable.attributes["type"] = "image"
                image_variable.attributes["origin"] = str(filename.parent)
                purged_sample[key] = str(filename)
                variables.append(image_variable)

        processed_attributes = []
        structured_dict = XConfig.from_dict(sample)
        for chunk in structured_dict.chunks():
            variable = self.DATA_REMAP[type(chunk[1])](chunk[0])
            variables.append(variable)
            # processed_attributes.append(self.DATA_REMAP[type(chunk[1])](chunk[0]))
            # processed_attributes.append(StringVariable.make(chunk[0]))
            purged_sample[chunk[0]] = chunk[1]

        processed_attributes = [
            x for x in variables if not isinstance(x, StringVariable)
        ]
        processed_meta = [x for x in variables if isinstance(x, StringVariable)]
        return {
            "attributes": processed_attributes,
            "metas": processed_meta,
            "purged_sample": purged_sample,
        }


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(EyecanUnderfolderReader).run()
