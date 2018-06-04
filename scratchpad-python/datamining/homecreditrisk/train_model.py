import sys

from datamining.homecreditrisk.library.Setup import Setup
from sklearn import preprocessing, metrics
import pandas as pd

path_setup_json = sys.argv[1]
try:
    max_epoch_run = int(sys.argv[2])
except IndexError as e:
    max_epoch_run = None
except Exception as e:
    max_epoch_run = 5

continue_setup = Setup('')
continue_setup.load(rel_filepath=path_setup_json)
expected_epochs = continue_setup.getOthers()['expected_epochs']
actual_epochs = continue_setup.getEpoch()
status_running = continue_setup.getOthers()['running'] if 'running' in continue_setup.getOthers() else False

setup_save_path = 'setup'

if actual_epochs < expected_epochs and not status_running:
    training_ids, training_data, training_targets, \
        validation_ids, validation_data, validation_targets, \
        testing_ids, testing_data, testing_targets = continue_setup.getData()

    training_targets_onehot = (preprocessing.OneHotEncoder().fit_transform(training_targets.reshape(-1, 1))).toarray()
    validation_targets_onehot = (preprocessing.OneHotEncoder().fit_transform(validation_targets.reshape(-1, 1))).toarray()

    continue_setup.setOthers({'running': True})
    continue_setup.save_setupfile(setup_save_path)

    epoch_run = 0
    for epoch in range(continue_setup.getEpoch() + 1, expected_epochs + 1):
        print('Training \'%s\': Epoch %d' % (continue_setup.getName(), epoch))
        dropout = continue_setup.getModel().fit(training_data, training_targets_onehot,
                                                batch_size=1000, epochs=1, verbose=True,
                                                validation_data=(validation_data, validation_targets_onehot))

        training_predictions_onehot = continue_setup.getModel().predict(training_data)
        validation_predictions_onehot = continue_setup.getModel().predict(validation_data)

        training_predictions = pd.DataFrame(training_predictions_onehot).apply(lambda val: 1.0 if val[1] > 0.50 else 0.0, axis=1)
        validation_predictions = pd.DataFrame(validation_predictions_onehot).apply(lambda val: 1.0 if val[1] > 0.50 else 0.0, axis=1)

        continue_setup.updateEpochs(add_epochs=1,
                                    training_acc=[metrics.accuracy_score(training_targets, training_predictions)],
                                    training_auc=[metrics.roc_auc_score(training_targets, training_predictions)],
                                    training_loss=[metrics.mean_squared_error(training_targets, training_predictions)],
                                    validation_acc=[metrics.accuracy_score(validation_targets, validation_predictions)],
                                    validation_auc=[metrics.roc_auc_score(validation_targets, validation_predictions)],
                                    validation_loss=[metrics.mean_squared_error(validation_targets, validation_predictions)],
                                    testing_acc=[0],
                                    testing_auc=[0],
                                    testing_loss=[0],
                                    allow_modify=True)

        continue_setup.save('setup')

        epoch_run = epoch_run + 1

        if max_epoch_run is not None and epoch_run > max_epoch_run:
            break

    continue_setup.setOthers({'running': False})
    continue_setup.save_setupfile(setup_save_path)

else:
    print('Actual Epochs: %d. Expected Epochs: %d' % (actual_epochs, expected_epochs))
