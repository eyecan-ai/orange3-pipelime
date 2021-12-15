from AnyQt.QtWidgets import QLabel
from Orange.widgets.widget import OWWidget, Output, Input
import Orange
import torch
from torchvision.models.resnet import resnet18


class EyeExtractor(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "FeatureExtractor"
    icon = "icons/mywidget.svg"
    want_main_area = False

    class Inputs:
        images = Input("images", Orange.data.Table)

    class Outputs:
        output = Output("outputx", int)

    def __init__(self):
        super().__init__()

        self._model = resnet18(pretrained=True)
        self._images = None
        label = QLabel("Hello, World!")
        self.controlArea.layout().addWidget(label)

    @Inputs.images
    def set_Images(self, images):
        self._images = images

    def handleNewSignals(self):
        if self._images is not None:
            self.Outputs.output.send(1)
        else:
            self.Outputs.output.send(0)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(EyeExtractor).run()
