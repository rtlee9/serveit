"""Base ModelServer test class."""
import json
import numpy as np

from serveit.server import ModelServer


class ModelServerTest(object):
    """Base class to test the prediction server.

    ModelServerTest should be inherited by a class that has a `model` attribute,
    and calls `ModelServerTest._setup()` after instantiation. That class should
    also inherit from `unittest.TestCase` to ensure tests are executed.
    """

    def _setup(self, model, fit, data, predict=None, **kwargs):
        """Set up method to be called before each unit test.

        Arguments:
            - fit (callable): model training method; must accept args (data, target)
        """
        self.data = data
        fit(self.data.data, self.data.target)
        self.predict = predict or self.model.predict
        self.server_kwargs = kwargs
        self.server = ModelServer(self.model, self.predict, **kwargs)
        self.app = self.server.app.test_client()

    @staticmethod
    def _prediction_post(app, data):
        """Make a POST request to `app` with JSON body `data`."""
        return app.post(
            '/predictions',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(data),
        )

    def _get_sample_data(self, n=100):
        """Return a sample of size n of self.data."""
        sample_idx = np.random.randint(self.data.data.shape[0], size=n)
        return self.data.data[sample_idx, :]

    def test_404_media(self):
        """Make sure API serves 404 response with JSON."""
        response = self.app.get('/fake-endpoint')
        self.assertEqual(response.status_code, 404)
        response_data_raw = response.get_data()
        self.assertIsNotNone(response_data_raw)
        response_data = json.loads(response_data_raw)
        self.assertGreater(len(response_data), 0)

    def test_features_info_none(self):
        """Verify 404 response if '/info/features' endpoint not yet created."""
        response = self.app.get('/info/features')
        self.assertEqual(response.status_code, 404)

    def test_features_info(self):
        """Test features info endpoint."""
        self.server.create_info_endpoint('features', self.data.feature_names)
        app = self.server.app.test_client()
        response = app.get('/info/features')
        response_data = json.loads(response.get_data())
        self.assertEqual(len(response_data), self.data.data.shape[1])
        try:
            self.assertCountEqual(response_data, self.data.feature_names)
        except AttributeError:  # Python 2
            self.assertItemsEqual(response_data, self.data.feature_names)

    def test_target_labels_info_none(self):
        """Verify 404 response if '/info/target_labels' endpoint not yet created."""
        response = self.app.get('/info/target_labels')
        self.assertEqual(response.status_code, 404)

    def test_target_labels_info(self):
        """Test target labels info endpoint."""
        if not hasattr(self.data, 'target_names'):
            return
        self.server.create_info_endpoint('target_labels', self.data.target_names.tolist())
        app = self.server.app.test_client()
        response = app.get('/info/target_labels')
        response_data = json.loads(response.get_data())
        self.assertEqual(len(response_data), self.data.target_names.shape[0])
        try:
            self.assertCountEqual(response_data, self.data.target_names)
        except AttributeError:  # Python 2
            self.assertItemsEqual(response_data, self.data.target_names)

    def test_predictions(self):
        """Test predictions endpoint."""
        sample_data = self._get_sample_data()
        response = self._prediction_post(self.app, sample_data.tolist())
        response_data = json.loads(response.get_data())
        self.assertEqual(len(response_data), len(sample_data))
        if self.data.target.ndim > 1:
            # for multiclass each prediction should be one of the training labels
            for prediction in response_data:
                self.assertIn(prediction, self.data.target)
        else:
            # the average regression prediction for a sample of data should be similar
            # to the population mean
            # TODO: remove variance from this test (i.e., no chance of false negative)
            pred_pct_diff = np.array(response_data).mean() / self.data.target.mean() - 1
            self.assertAlmostEqual(pred_pct_diff / 1e4, 0, places=1)

    def test_input_validation(self):
        """Add simple input validator and make sure it triggers."""
        # model input validator
        def feature_count_check(data):
            try:
                # convert PyTorch variables to numpy arrays
                data = data.data.numpy()
            except:
                pass
            # check num dims
            if data.ndim != 2:
                return False, 'Data should have two dimensions.'
            # check number of columns
            if data.shape[1] != self.data.data.shape[1]:
                reason = '{} features required, {} features provided'.format(
                    data.shape[1], self.data.data.shape[1])
                return False, reason
            # validation passed
            return True, None

        # set up test server
        server = ModelServer(self.model, self.predict, feature_count_check, **self.server_kwargs)
        app = server.app.test_client()

        # generate sample data
        sample_data = self._get_sample_data()

        # post good data, verify 200 response
        response = self._prediction_post(app, sample_data.tolist())
        self.assertEqual(response.status_code, 200)

        # post bad data (drop a single column), verify 400 response
        response = self._prediction_post(app, sample_data[:, :-1].tolist())
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.get_data())
        expected_reason = '{} features required, {} features provided'.format(
            self.data.data.shape[1] - 1, self.data.data.shape[1])
        self.assertIn(expected_reason, response_data['message'])

    def test_model_info(self):
        """Test model info endpoint."""
        response = self.app.get('/info/model')
        response_data = json.loads(response.get_data())
        self.assertGreater(len(response_data), 3)  # TODO: expand test scope

    def test_data_loader(self):
        """Test model prediction with a custom data loader callback."""
        # TODO: test alternative request method (e.g., URL params)
        # define custom data loader
        def read_json_from_dict():
            from flask import request
            # read data as the value of the 'data' key
            data = request.get_json()
            return np.array(data['data'])

        # create test client
        server = ModelServer(self.model, self.predict, data_loader=read_json_from_dict, **self.server_kwargs)
        app = server.app.test_client()

        # generate sample data, and wrap in dict keyed by 'data'
        sample_data = self._get_sample_data()
        data_dict = dict(data=sample_data.tolist())

        response = self._prediction_post(app, data_dict)
        response_data = json.loads(response.get_data())
        self.assertEqual(len(response_data), len(sample_data))
        if self.data.target.ndim > 1:
            # for multiclass each prediction should be one of the training labels
            for prediction in response_data:
                self.assertIn(prediction, self.data.target)
        else:
            # the average regression prediction for a sample of data should be similar
            # to the population mean
            # TODO: remove variance from this test (i.e., no chance of false negative)
            pred_pct_diff = np.array(response_data).mean() / self.data.target.mean() - 1
            self.assertAlmostEqual(pred_pct_diff / 1e4, 0, places=1)

    def _update_kwargs_item(self, item, key_name, position='first'):
        """Prepend a method to the existing preprocessing chain, add to self's kwargs and return."""
        kwargs = self.server_kwargs
        if key_name in self.server_kwargs:
            existing_items = kwargs[key_name]
            if not isinstance(existing_items, (list, tuple)):
                existing_items = [existing_items]
        else:
            existing_items = []
        if position == 'first':
            kwargs[key_name] = [item] + existing_items
        if position == 'last':
            kwargs[key_name] = existing_items + [item]
        return kwargs

    def test_preprocessing(self):
        """Test predictions endpoint with custom preprocessing callback."""
        # create test client with postprocessor that unraps data from a dict as the value of the 'data' key
        kwargs = self._update_kwargs_item(lambda d: d['data'], 'preprocessor')
        server = ModelServer(self.model, self.predict, **kwargs)
        app = server.app.test_client()

        # generate sample data, and wrap in dict keyed by 'data'
        sample_data = self._get_sample_data()
        data_dict = dict(data=sample_data.tolist())

        response = self._prediction_post(app, data_dict)
        response_data = json.loads(response.get_data())
        self.assertEqual(len(response_data), len(sample_data))
        if self.data.target.ndim > 1:
            # for multiclass each prediction should be one of the training labels
            for prediction in response_data:
                self.assertIn(prediction, self.data.target)
        else:
            # the average regression prediction for a sample of data should be similar
            # to the population mean
            # TODO: remove variance from this test (i.e., no chance of false negative)
            pred_pct_diff = np.array(response_data).mean() / self.data.target.mean() - 1
            self.assertAlmostEqual(pred_pct_diff / 1e4, 0, places=1)

    def test_preprocessing_list(self):
        """Test predictions endpoint with chained preprocessing callbacks."""
        # create test client with postprocessor that unraps data from a dict as the value of the 'data' key
        kwargs = self._update_kwargs_item(lambda d: d['data'], 'preprocessor')
        kwargs['preprocessor'] = [lambda d: d['data2']] + kwargs['preprocessor']
        server = ModelServer(
            self.model,
            self.predict,
            **kwargs
        )
        app = server.app.test_client()

        # generate sample data, and wrap in dict keyed by 'data'
        sample_data = self._get_sample_data()
        data_dict = dict(data2=dict(data=sample_data.tolist()))

        response = self._prediction_post(app, data_dict)
        response_data = json.loads(response.get_data())
        self.assertEqual(len(response_data), len(sample_data))
        if self.data.target.ndim > 1:
            # for multiclass each prediction should be one of the training labels
            for prediction in response_data:
                self.assertIn(prediction, self.data.target)
        else:
            # the average regression prediction for a sample of data should be similar
            # to the population mean
            # TODO: remove variance from this test (i.e., no chance of false negative)
            pred_pct_diff = np.array(response_data).mean() / self.data.target.mean() - 1
            self.assertAlmostEqual(pred_pct_diff / 1e4, 0, places=1)

    def test_postprocessing(self):
        """Test predictions endpoint with custom postprocessing callback."""
        # create test client with postprocessor that wraps predictions in a dictionary
        kwargs = self._update_kwargs_item(lambda x: dict(prediction=x.tolist()), 'postprocessor', 'last')
        server = ModelServer(self.model, self.predict, **kwargs)
        app = server.app.test_client()

        # generate sample data
        sample_data = self._get_sample_data()

        response = self._prediction_post(app, sample_data.tolist())
        response_data = json.loads(response.get_data())['prediction']  # predictions are nested under 'prediction' key
        self.assertEqual(len(response_data), len(sample_data))
        if self.data.target.ndim > 1:
            # for multiclass each prediction should be one of the training labels
            for prediction in response_data:
                self.assertIn(prediction, self.data.target)
        else:
            # the average regression prediction for a sample of data should be similar
            # to the population mean
            # TODO: remove variance from this test (i.e., no chance of false negative)
            pred_pct_diff = np.array(response_data).mean() / self.data.target.mean() - 1
            self.assertAlmostEqual(pred_pct_diff / 1e4, 0, places=1)

    def test_get_app(self):
        """Make sure get_app method returns the same app."""
        self.assertEqual(self.server.get_app(), self.server.app)

    def test_400_no_content_type(self):
        """Check 400 response if no Content-Type header specified."""
        response = self.app.post(
            '/predictions',
        )
        self.assertEqual(response.status_code, 400)
        response_body = json.loads(response.get_data())
        self.assertEqual(response_body['message'], 'Unable to fetch data')
        self.assertGreaterEqual(len(response_body['details']), 2)
