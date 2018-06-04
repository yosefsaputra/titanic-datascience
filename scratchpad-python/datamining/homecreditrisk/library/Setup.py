import warnings

import numpy as np
import json
import os
import datetime
import glob
import re
import ntpath
import shutil

from keras.models import load_model


class Setup(object):
    def __init__(self, name):
        self._name = name
        self._model = None
        self._emptyModel = None

        self._training_ids = None
        self._validation_ids = None
        self._testing_ids = None
        self._training_data = None
        self._validation_data = None
        self._testing_data = None
        self._training_targets = None
        self._validation_targets = None
        self._testing_targets = None

        self._training_ids_directory = None
        self._validation_ids_directory = None
        self._testing_ids_directory = None
        self._training_data_directory = None
        self._validation_data_directory = None
        self._testing_data_directory = None
        self._training_targets_directory = None
        self._validation_targets_directory = None
        self._testing_targets_directory = None

        self._training_accuracy = []
        self._training_auc = []
        self._training_loss = []
        self._validation_accuracy = []
        self._validation_auc = []
        self._validation_loss = []
        self._testing_accuracy = []
        self._testing_auc = []
        self._testing_loss = []

        self._batch_size = None
        self._epochs = 0

        self._others = {}

        self._setup = {
            'name': self._name,

            'file': {
                'setup': '',
                'model': '',
                'model_arch_json': '',
                'model_arch_yaml': '',
                'model_weights': '',

                'training_ids': '',
                'validation_ids': '',
                'testing_ids': '',
                'training_data': '',
                'validation_data': '',
                'testing_data': '',
                'training_targets': '',
                'validation_targets': '',
                'testing_targets': '',
            },

            'directory': {
                'training_ids': '',
                'validation_ids': '',
                'testing_ids': '',
                'training_data': '',
                'validation_data': '',
                'testing_data': '',
                'training_targets': '',
                'validation_targets': '',
                'testing_targets': '',
            },

            'training_accuracy': self._training_accuracy,
            'training_accuracy': self._training_auc,
            'training_loss': self._training_loss,
            'validation_accuracy': self._validation_accuracy,
            'validation_accuracy': self._validation_auc,
            'validation_loss': self._validation_loss,
            'testing_accuracy': self._testing_accuracy,
            'testing_accuracy': self._testing_auc,
            'testing_loss': self._testing_loss,
            'epochs': self._epochs,
            'others': self._others,
        }

    def getName(self):
        return self._name

    def setName(self, name):
        self._name = name

    def getModel(self):
        return self._model

    def setModel(self, model):
        self._model = model
        # TODO: get the empty model and assign it to self._emptyModel

    def getData(self):
        return self._training_ids, self._training_data, self._training_targets, \
               self._validation_ids, self._validation_data, self._validation_targets, \
               self._testing_ids, self._testing_data, self._testing_targets

    def setData(self,
                training_ids=None, training_data=None, training_targets=None,
                validation_ids=None, validation_data=None, validation_targets=None,
                testing_ids=None, testing_data=None, testing_targets=None):
        self._training_ids = training_ids if training_ids is not None else self._training_ids
        self._validation_ids = validation_ids if validation_ids is not None else self._validation_ids
        self._testing_ids = testing_ids if testing_ids is not None else self._testing_ids

        self._training_data = training_data if training_data is not None else self._training_data
        self._validation_data = validation_data if validation_data is not None else self._validation_data
        self._testing_data = testing_data if testing_data is not None else self._testing_data

        self._training_targets = training_targets if training_targets is not None else self._training_targets
        self._validation_targets = validation_targets if validation_targets is not None else self._validation_targets
        self._testing_targets = testing_targets if testing_targets is not None else self._testing_targets

    def getDataDirectory(self):
        return self._training_data_directory, self._training_data_directory, self._training_targets_directory, \
               self._validation_data_directory, self._validation_data_directory, self._validation_targets_directory, \
               self._testing_data_directory, self._testing_data_directory, self._testing_targets_directory

    def setDataDirectory(self,
                         training_ids_directory=None, training_data_directory=None, training_targets_directory=None,
                         validation_ids_directory=None, validation_data_directory=None, validation_targets_directory=None,
                         testing_ids_directory=None, testing_data_directory=None, testing_targets_directory=None):

        self._training_ids_directory = training_ids_directory if training_ids_directory is not None else self._training_ids_directory
        self._validation_ids_directory = validation_ids_directory if validation_ids_directory is not None else self._validation_ids_directory
        self._testing_ids_directory = testing_ids_directory if testing_ids_directory is not None else self._testing_ids_directory

        self._training_data_directory = training_data_directory if training_data_directory is not None else self._training_data_directory
        self._validation_data_directory = validation_data_directory if validation_data_directory is not None else self._validation_data_directory
        self._testing_data_directory = testing_data_directory if testing_data_directory is not None else self._testing_data_directory

        self._training_targets_directory = training_targets_directory if training_targets_directory is not None else self._training_targets_directory
        self._validation_targets_directory = validation_targets_directory if validation_targets_directory is not None else self._validation_targets_directory
        self._testing_targets_directory = testing_targets_directory if testing_targets_directory is not None else self._testing_targets_directory

    def getEpoch(self):
        return self._epochs

    def updateEpochs(self, add_epochs,
                     training_acc, training_auc, training_loss,
                     validation_acc, validation_auc, validation_loss,
                     testing_acc, testing_auc, testing_loss,
                     allow_modify=True):
        # TODO: check

        def checkListLength(length, mList, listName, allowModify):
            modifiedList = mList

            if allow_modify:
                if len(modifiedList) < length:
                    modifiedList.extend([mList[-1] for i in range(length - len(modifiedList))])
                elif len(modifiedList) > length:
                    warnings.warn('%s list is longer than add_epochs. Trimmed list will be used.' % listName)
                    modifiedList = modifiedList[:length]
            else:
                if len(modifiedList) != length:
                    raise ValueError('%s list length is not equal to add_epochs' % listName)

            return modifiedList

        # Checking parameters
        # add_epochs
        if add_epochs is None or type(add_epochs) != int:
            raise TypeError('add_epochs should have type \'int\'')
        elif add_epochs < 0:
            raise ValueError('add_epochs should be > 0')

        if training_acc is None or type(training_acc) != list or \
                training_auc is None or type(training_auc) != list or \
                training_loss is None or type(training_loss) != list or \
                validation_acc is None or type(validation_acc) != list or \
                validation_auc is None or type(validation_auc) != list or \
                validation_loss is None or type(validation_loss) != list or \
                testing_acc is None or type(testing_acc) != list or \
                testing_auc is None or type(testing_auc) != list or \
                testing_loss is None or type(testing_loss) != list:
            raise TypeError('training_acc, training_auc, training_loss, '
                            'validation_acc, validation_auc, validation_loss, '
                            'testing_acc, testing_auc, testing_loss should have type \'list\'')

        new_train_acc = checkListLength(add_epochs, training_acc, 'training_acc', allow_modify)
        new_train_auc = checkListLength(add_epochs, training_auc, 'training_auc', allow_modify)
        new_train_loss = checkListLength(add_epochs, training_loss, 'training_loss', allow_modify)
        new_val_acc = checkListLength(add_epochs, validation_acc, 'validation_acc', allow_modify)
        new_val_auc = checkListLength(add_epochs, validation_auc, 'validation_auc', allow_modify)
        new_val_loss = checkListLength(add_epochs, validation_loss, 'validation_loss', allow_modify)
        new_test_acc = checkListLength(add_epochs, testing_acc, 'testing_acc', allow_modify)
        new_test_auc = checkListLength(add_epochs, testing_auc, 'testing_auc', allow_modify)
        new_test_loss = checkListLength(add_epochs, testing_loss, 'testing_loss', allow_modify)

        self._epochs += add_epochs
        self._training_accuracy.extend(new_train_acc)
        self._training_auc.extend(new_train_auc)
        self._training_loss.extend(new_train_loss)
        self._validation_accuracy.extend(new_val_acc)
        self._validation_auc.extend(new_val_auc)
        self._validation_loss.extend(new_val_loss)
        self._testing_accuracy.extend(new_test_acc)
        self._testing_auc.extend(new_test_auc)
        self._testing_loss.extend(new_test_loss)

    def getOthers(self):
        return self._others

    def setOthers(self, others):
        for key in others:
            self._others[key] = others[key]

    def save(self, rel_path):
        # Save every information or object

        if rel_path is None:
            raise ValueError('rel_path should not be None')
        else:
            pass

        self._setup['name'] = self._name
        self._setup['time'] = str(datetime.datetime.now()),

        self._setup['file']['setup'] = os.path.join('setup.json')
        self._setup['file']['model'] = os.path.join('model.h5')
        self._setup['file']['model_arch_json'] = os.path.join('model_architecture.json')
        self._setup['file']['model_arch_yaml'] = os.path.join('model_architecture.yaml')
        self._setup['file']['model_weights'] = os.path.join('model_weights.h5')

        self._setup['file']['training_ids'] = os.path.join('training_ids.npy')
        self._setup['file']['validation_ids'] = os.path.join('validation_ids.npy')
        self._setup['file']['testing_ids'] = os.path.join('testing_ids.npy')
        self._setup['file']['training_data'] = os.path.join('training_data.npy')
        self._setup['file']['validation_data'] = os.path.join('validation_data.npy')
        self._setup['file']['testing_data'] = os.path.join('testing_data.npy')
        self._setup['file']['training_targets'] = os.path.join('training_targets.npy')
        self._setup['file']['validation_targets'] = os.path.join('validation_targets.npy')
        self._setup['file']['testing_targets'] = os.path.join('testing_targets.npy')

        self._setup['directory']['training_ids'] = self._training_ids_directory
        self._setup['directory']['validation_ids'] = self._validation_ids_directory
        self._setup['directory']['testing_ids'] = self._testing_ids_directory
        self._setup['directory']['training_data'] = self._training_data_directory
        self._setup['directory']['validation_data'] = self._validation_data_directory
        self._setup['directory']['testing_data'] = self._testing_data_directory
        self._setup['directory']['training_targets'] = self._training_targets_directory
        self._setup['directory']['validation_targets'] = self._validation_targets_directory
        self._setup['directory']['testing_targets'] = self._testing_targets_directory

        self._setup['training_accuracy'] = self._training_accuracy
        self._setup['training_auc'] = self._training_auc
        self._setup['training_loss'] = self._training_loss
        self._setup['validation_accuracy'] = self._validation_accuracy
        self._setup['validation_auc'] = self._validation_auc
        self._setup['validation_loss'] = self._validation_loss
        self._setup['testing_accuracy'] = self._testing_accuracy
        self._setup['testing_auc'] = self._testing_auc
        self._setup['testing_loss'] = self._testing_loss
        self._setup['epochs'] = self._epochs

        self._setup['others'] = self._others

        if not os.path.exists(os.path.join(os.getcwd(), rel_path)):
            os.mkdir(os.path.join(os.getcwd(), rel_path))

        if not os.path.exists(os.path.join(os.getcwd(), rel_path, self._name)):
            os.mkdir(os.path.join(os.getcwd(), rel_path, self._name))

        if len(glob.glob(os.path.join(os.getcwd(), rel_path, self._name, '*.*'))) > 0:
            versions = []
            pattern = r'^.*version(?P<versionnumber>\d*)$'
            for dir in glob.glob(os.path.join(os.getcwd(), rel_path, self._name, 'version*')):
                regex = re.search(pattern, dir)
                versions.append(int(regex.group('versionnumber')))
            if len(versions) == 0:
                maxVer = 0
            else:
                maxVer = np.max(versions)

            newVerDirName = 'version%s' % (maxVer + 1)
            os.mkdir(os.path.join(os.getcwd(), rel_path, self._name, newVerDirName))
            self._backup_version(os.path.join(os.getcwd(), rel_path, self._name),
                                 os.path.join(os.getcwd(), rel_path, self._name, newVerDirName))

            if (maxVer + 1 - 10) > 0 and (maxVer + 1 - 10) % 20 != 0:
                oldVerDirName = 'version%s' % (maxVer + 1 - 10)
                shutil.rmtree(os.path.join(os.getcwd(), rel_path, self._name, oldVerDirName))

        # ==========================================
        # Save whole model
        self._model.save(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['model']))

        # ==========================================
        # Save model architecture
        json_model_arch = self._model.to_json()
        with open(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['model_arch_json']), 'w') as jsonfile:
            jsonfile.write(json_model_arch)

        yaml_model_arch = self._model.to_yaml()
        with open(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['model_arch_yaml']), 'w') as yamlfile:
            yamlfile.write(yaml_model_arch)

        # ==========================================
        # Save model weights
        self._model.save_weights(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['model_weights']))

        # ==========================================
        # Save data
        if self._training_ids is not None and type(self._training_ids) == np.ndarray:
            try:
                np.save(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['training_ids']), self._training_ids)
            except Exception as e:
                self._setup['file']['training_ids'] = None
        if self._validation_ids is not None and type(self._validation_ids) == np.ndarray:
            try:
                np.save(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['validation_ids']), self._validation_ids)
            except Exception as e:
                self._setup['file']['validation_ids'] = None
        if self._testing_ids is not None and type(self._testing_ids) == np.ndarray:
            try:
                np.save(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['testing_ids']), self._testing_ids)
            except Exception as e:
                self._setup['file']['testing_ids'] = None
        if self._training_data is not None and type(self._training_data) == np.ndarray:
            try:
                np.save(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['training_data']), self._training_data)
            except Exception as e:
                self._setup['file']['training_data'] = None
        if self._validation_data is not None and type(self._validation_data) == np.ndarray:
            try:
                np.save(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['validation_data']), self._validation_data)
            except Exception as e:
                self._setup['file']['validation_data'] = None
        if self._testing_data is not None and type(self._testing_data) == np.ndarray:
            try:
                np.save(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['testing_data']), self._testing_data)
            except Exception as e:
                self._setup['file']['testing_data'] = None
        if self._training_targets is not None and type(self._training_targets) == np.ndarray:
            try:
                np.save(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['training_targets']), self._training_targets)
            except Exception as e:
                self._setup['file']['training_targets'] = None
        if self._validation_targets is not None and type(self._validation_targets) == np.ndarray:
            try:
                np.save(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['validation_targets']), self._validation_targets)
            except Exception as e:
                self._setup['file']['validation_targets'] = None
        if self._testing_targets is not None and type(self._testing_targets) == np.ndarray:
            try:
                np.save(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['testing_targets']), self._testing_targets)
            except Exception as e:
                self._setup['file']['testing_targets'] = None

        # ==========================================
        # Save setup
        with open(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['setup']), 'w') as setupfile:
            json.dump(self._setup, setupfile)

    def load(self, rel_filepath):
        cwd = os.getcwd()

        if rel_filepath is None:
            raise ValueError('rel_filepath should be None')
        else:
            pass

        # ==========================================
        # Load info
        with open(os.path.join(cwd, rel_filepath), 'r') as setupfile:
            self._setup = json.load(setupfile)

        rel_filepath = rel_filepath.replace('setup.json', '')

        # ==========================================
        # Load name
        self._name = self._setup['name']

        # ==========================================
        # Load whole model
        self._model = load_model(os.path.join(cwd, rel_filepath, self._setup['file']['model']))

        # TODO: if loading from model h5 file fails, then load from model arch file and load weights
        # # ==========================================
        # # Load model architecture
        # with open(os.path.join(directory, self.setup['model_arch_json']), 'r') as jsonfile:
        #     self.emptyModel = model_from_json(jsonfile.read())
        #
        # with open(os.path.join(directory, self.setup['model_arch_yaml']), 'r') as yamlfile:
        #     self.emptyModel = model_from_yaml(yamlfile.read())

        # # ==========================================
        # # Load model weights
        # self.model.load_weights(os.path.join(directory, self.setup['model_weights']))

        # ==========================================
        # Load data
        self._training_ids = np.load(os.path.join(cwd, rel_filepath, self._setup['file']['training_ids'])) \
            if os.path.exists(os.path.join(cwd, rel_filepath, self._setup['file']['training_ids'])) else self._training_ids
        self._validation_ids = np.load(os.path.join(cwd, rel_filepath, self._setup['file']['validation_ids'])) \
            if os.path.exists(os.path.join(cwd, rel_filepath, self._setup['file']['validation_ids'])) else self._validation_ids
        self._testing_ids = np.load(os.path.join(cwd, rel_filepath, self._setup['file']['testing_ids'])) \
            if os.path.exists(os.path.join(cwd, rel_filepath, self._setup['file']['testing_ids'])) else self._testing_ids
        self._training_data = np.load(os.path.join(cwd, rel_filepath, self._setup['file']['training_data'])) \
            if os.path.exists(os.path.join(cwd, rel_filepath, self._setup['file']['training_data'])) else self._training_data
        self._validation_data = np.load(os.path.join(cwd, rel_filepath, self._setup['file']['validation_data'])) \
            if os.path.exists(os.path.join(cwd, rel_filepath, self._setup['file']['validation_data'])) else self._validation_data
        self._testing_data = np.load(os.path.join(cwd, rel_filepath, self._setup['file']['testing_data'])) \
            if os.path.exists(os.path.join(cwd, rel_filepath, self._setup['file']['testing_data'])) else self._testing_data
        self._training_targets = np.load(os.path.join(cwd, rel_filepath, self._setup['file']['training_targets'])) \
            if os.path.exists(os.path.join(cwd, rel_filepath, self._setup['file']['training_targets'])) else self._training_targets
        self._validation_targets = np.load(os.path.join(cwd, rel_filepath, self._setup['file']['validation_targets'])) \
            if os.path.exists(os.path.join(cwd, rel_filepath, self._setup['file']['validation_targets'])) else self._validation_targets
        self._testing_targets = np.load(os.path.join(cwd, rel_filepath, self._setup['file']['testing_targets'])) \
            if os.path.exists(os.path.join(cwd, rel_filepath, self._setup['file']['testing_targets'])) else self._testing_targets

        # ==========================================
        # Load data directory
        self._training_data_directory = self._setup['directory']['training_data']
        self._validation_data_directory = self._setup['directory']['validation_data']
        self._testing_data_directory = self._setup['directory']['testing_data']
        self._training_targets_directory = self._setup['directory']['training_targets']
        self._validation_targets_directory = self._setup['directory']['validation_targets']
        self._testing_targets_directory = self._setup['directory']['testing_targets']

        # ==========================================
        # Load info
        self._training_accuracy = self._setup['training_accuracy']
        self._training_auc = self._setup['training_auc']
        self._training_loss = self._setup['training_loss']
        self._validation_accuracy = self._setup['validation_accuracy']
        self._validation_auc = self._setup['validation_auc']
        self._validation_loss = self._setup['validation_loss']
        self._testing_accuracy = self._setup['testing_accuracy']
        self._testing_auc = self._setup['testing_auc']
        self._testing_loss = self._setup['testing_loss']
        self._epochs = self._setup['epochs']

        self._others = self._setup['others']

    def save_setupfile(self, rel_path):
        with open(os.path.join(os.getcwd(), rel_path, self._name, self._setup['file']['setup']), 'w') as setupfile:
            json.dump(self._setup, setupfile)

    def _backup_version(self, source, destination):
        ignore_list = ['training_data',
                       'validation_data',
                       'testing_data',
                       'training_targets',
                       'validation_targets',
                       'testing_targets']
        for file in glob.glob(os.path.join(source, '*.*')):
            ignore = False
            for ignored_filename in ignore_list:
                if ignored_filename in file:
                    ignore = True

            if ignore:
                continue
            else:
                os.rename(file, os.path.join(destination, ntpath.basename(file)))
