from torch import nn


def model_from_name(name, outputlayer_size, n_sensors):
    if name == 'shallow_decoder':
        return ShallowDecoder(outputlayer_size, n_sensors)
    elif name == 'shallow_decoder_drop':
        return ShallowDecoderDrop(outputlayer_size, n_sensors)
    elif name == 'shallow_decoder_sensitivity':
        return ShallowDecoderSensitivity(outputlayer_size, n_sensors)
    elif name == 'shallow_decoder_sensitivity_small':
        return ShallowDecoderSensitivity_small(outputlayer_size, n_sensors)

    raise ValueError('model {} not recognized'.format(name))


class ShallowDecoder(nn.Module):
    def __init__(self, outputlayer_size, n_sensors):
        super(ShallowDecoder, self).__init__()

        self.n_sensors = n_sensors
        self.outputlayer_size = outputlayer_size

        self.learn_features = nn.Sequential(
            nn.Linear(n_sensors, 100),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
        )

        self.learn_coef = nn.Sequential(
            nn.Linear(100, 150),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
        )

        self.learn_dictionary = nn.Sequential(
            nn.Linear(150, self.outputlayer_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, x):
        x = self.learn_features(x)
        x = self.learn_coef(x)
        x = self.learn_dictionary(x)
        return x


class ShallowDecoderDrop(nn.Module):
    def __init__(self, outputlayer_size, n_sensors):
        super(ShallowDecoderDrop, self).__init__()

        self.n_sensors = n_sensors
        self.outputlayer_size = outputlayer_size

        self.learn_features = nn.Sequential(
            nn.Linear(n_sensors, 200),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
        )

        self.learn_coef = nn.Sequential(
            nn.Linear(200, 950),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
        )

        self.learn_dictionary = nn.Sequential(
            nn.Linear(950, self.outputlayer_size),
        )
        
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, x):
        x = self.learn_features(x)
        x = nn.functional.dropout(x, p=0.1, training=self.training)
        x = self.learn_coef(x)
        x = self.learn_dictionary(x)
        return x


class ShallowDecoderSensitivity(nn.Module):
    def __init__(self, outputlayer_size, n_sensors):
        super(ShallowDecoderSensitivity, self).__init__()

        #Same as shallow_decoder drop except removes bath normalization
        self.n_sensors = n_sensors
        self.outputlayer_size = outputlayer_size

        self.layers = nn.Sequential(
            nn.Linear(n_sensors, 200),
            nn.ReLU(True),
            # nn.BatchNorm1d(1),
            nn.Linear(200, 950),
            nn.ReLU(True),
            # nn.BatchNorm1d(1),
            nn.Linear(950, self.outputlayer_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0.0)


class ShallowDecoderSensitivity_small(nn.Module):
    def __init__(self, outputlayer_size, n_sensors):
        super(ShallowDecoderSensitivity_small, self).__init__()

        #Same as shallow_decoder drop except removes bath normalization
        self.n_sensors = n_sensors
        self.outputlayer_size = outputlayer_size

        self.layers = nn.Sequential(
            nn.Linear(n_sensors, 50),
            nn.ReLU(True),
            # nn.BatchNorm1d(1),
            nn.Linear(50, 100),
            nn.ReLU(True),
            # nn.BatchNorm1d(1),
            nn.Linear(100, self.outputlayer_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0.0)
    def forward(self, x):
        x = self.layers(x)
        return x
