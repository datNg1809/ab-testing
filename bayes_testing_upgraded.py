import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
import db_cred as db
import os


class Bayesian_AB:
    '''
    '''

    def __init__(self,
                 project_name=None,
                 control_filter='old_style_control',
                 variant_filter='new_style_variant',
                 date_column='EVENT_DAY',
                 sample_column='USER_ID',
                 conversion_column='LEADS',
                 test_column='TEST_GROUP',
                 alpha_prior=1,
                 beta_prior=1,
                 day_index=True,
                 simulations=1
                 ):
        self.project_name = project_name
        self.control_filter = control_filter
        self.variant_filter = variant_filter
        self.date_column = date_column
        self.sample_column = sample_column
        self.conversion_column = conversion_column
        self.test_column = test_column
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.day_index = day_index
        self.simulations = simulations
        self.results = None
        self.control_sample_size = None
        self.control_conversions = None
        self.variant_sample_size = None
        self.variant_conversions = None

    def __str__(self):
        word = '''
        Bayesian approach to AB Test.
        Credited to Marc Perez & Dat Nguyen from Analytics Team xD
        '''
        print(word)

    def get_data(self,
                 load_new: bool = True,
                 save_to_disc: bool = True,
                 query=None):

        if not load_new:
            df = self.load_data_from_disc()
        else:
            df = db.import_sf_sql(query)
            if save_to_disc:
                file_name = self.make_folder()
                df.to_csv(file_name, sep=',', index=False, compression='gzip')
        return df

    def load_data_from_disc(self):
        file_name = self.make_folder()
        df = pd.read_csv(file_name, sep=',', index_col=False, compression='gzip')
        return df

    def make_folder(self):
        path = os.environ['FILE_PATH']
        file_path = os.path.join(path, self.project_name)
        os.makedirs(file_path, exist_ok=True)
        file_name = (str(file_path) + '/' + str(self.project_name) + '.gz')
        return file_name

    def make_day_index(self, df, date_column):
        """creates index for dataframe, names it as DAY"""
        if type(df) is not pd.core.frame.DataFrame:
            raise TypeError('Argument is not Pandas DataFrame or is None')
        else:
            _index = pd.Series(np.arange(0, len(df), 1))
            df.sort_values(date_column, axis=0, inplace=True)
            df['DATE'] = df.index
            df['WEEK_DAY'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d').dt.day_name()
            df = df[df.columns[-2:].append(df.columns[0:2])]
            output = pd.DataFrame(df).set_index(_index)
            output.index.rename('DAY', inplace=True)
            return output

    def get_sequences_from_df(self, df: pd.DataFrame, n_column: str, k_column: str):
        def sequence(n: int, k: int):
            _n = int(n)
            _k = int(k)
            return np.concatenate([np.zeros(_n - _k), np.ones(_k)])

        sequences = np.concatenate(
            df.apply(lambda row: sequence(row[n_column], row[k_column]), axis=1).values)
        np.random.shuffle(sequences)
        return sequences

    def plot_lines(self,
                   control: pd.DataFrame,
                   variant: pd.DataFrame,
                   conversion_column=None,
                   sample_column=None
                   ):
        '''Plot daily'''
        if conversion_column is None:
            conversion_column = self.conversion_column
        if sample_column is None:
            sample_column = self.sample_column

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(control.index, control[conversion_column] / control[sample_column], label='control')
        ax.plot(variant.index, variant[conversion_column] / variant[sample_column], label='variant')
        ax.legend()
        ax.set_xlabel('Day')
        ax.set_title('Daily performance')
        plt.show()

    def transform_frame(self, df):
        # metric_order = ['uplift', 'control expected loss', 'variant expected loss', 'prob (variant >= control)', 'standard error']
        df = df.melt(id_vars='index', value_name='VALUE').set_index(['variable', 'index']).dropna()
        df.index.names = ['GROUP', 'METRIC']

        # new = df.reindex(level='metric', labels=metric_order)
        # df = df.loc[['control', 'variant']].append(new)
        return df

    def calc_min_interval(self, x, alpha):
        """Internal method to determine the minimum interval of a given width
        Assumes that x is sorted numpy array.
        """

        n = len(x)
        cred_mass = 1.0 - alpha
        x = np.sort(x)

        interval_idx_inc = int(np.floor(cred_mass * n))
        n_intervals = n - interval_idx_inc
        interval_width = x[interval_idx_inc:] - x[:n_intervals]

        if len(interval_width) == 0:
            raise ValueError('Too few elements for interval calculation')

        min_idx = np.argmin(interval_width)
        hdi_min = x[min_idx]
        hdi_max = x[min_idx + interval_idx_inc]
        return hdi_min, hdi_max

    def hpd(self, x, alpha=0.05):
        """Calculate the highest posterior density (HPD) of array for given alpha.
        The HPD is the minimum width Bayesian credible interval (BCI).
        :Arguments:
            x : Numpy array
            An array containing MCMC samples
            alpha : float
            Desired probability of type I error (defaults to 0.05)
        """
        # Make a copy of trace
        x = x.copy()
        # For multivariate node
        if x.ndim > 1:
            # Transpose first, then sort
            tx = np.transpose(x, list(range(x.ndim))[1:] + [0])
            dims = np.shape(tx)
            # Container list for intervals
            intervals = np.resize(0.0, dims[:-1] + (2,))

            for index in make_indices(dims[:-1]):
                try:
                    index = tuple(index)
                except TypeError:
                    pass

                # Sort trace
                sx = np.sort(tx[index])
                # Append to list
                intervals[index] = calc_min_interval(sx, alpha)
            # Transpose back before returning
            return np.array(intervals)
        else:
            # Sort univariate node
            sx = np.sort(x)
            return np.array(self.calc_min_interval(sx, alpha))

    def prepare_data_beta(self,
                          df: pd.DataFrame,
                          day_index: bool = True,
                          control_filter: str = None,
                          variant_filter: str = None,
                          date_column: str = None,
                          sample_column: str = None,
                          conversion_column: str = None,
                          test_column: str = None
                          ):
        if control_filter is None: control_filter = self.control_filter
        if variant_filter is None: variant_filter = self.variant_filter
        if date_column is None: date_column = self.date_column
        if sample_column is None: sample_column = self.sample_column
        if conversion_column is None: conversion_column = self.conversion_column
        if test_column is None: test_column = self.test_column

        df = df[[date_column, sample_column, test_column, conversion_column]]
        control_dataset = df[df[test_column] == control_filter].groupby(date_column).agg({sample_column: 'count',
                                                                                          conversion_column: 'sum'})
        variant_dataset = df[df[test_column] == variant_filter].groupby(date_column).agg({sample_column: 'count',
                                                                                          conversion_column: 'sum'})
        if day_index:
            c = self.make_day_index(control_dataset, date_column)
            v = self.make_day_index(variant_dataset, date_column)
        else:
            c = control_dataset
            v = variant_dataset

        # Override default values when declaring class
        self.control_filter = control_filter
        self.variant_filter = variant_filter
        self.date_column = date_column
        self.sample_column = sample_column
        self.conversion_column = conversion_column
        self.test_column = test_column
        self.control_sample_size = sum(c[sample_column])
        self.control_conversions = sum(c[conversion_column])
        self.variant_sample_size = sum(v[sample_column])
        self.variant_conversions = sum(v[conversion_column])

        return c, v

    def get_results_beta(self):
        results = self.results
        results = results.iloc[-1]

        uplift = (results['variant_cvr'] - results['control_cvr']) / results['control_cvr']

        output = {'control': {'CvR': '{:.5f}'.format(results['control_cvr']),
                              'sample size': self.control_sample_size,
                              'conversions': self.control_conversions,
                              '95% credible interval': ('{:.5f}'.format(results['control_cvr_lower']),
                                                        '{:.5f}'.format(results['control_cvr_upper']))},
                  'variant': {'CvR': '{:.5f}'.format(results['variant_cvr']),
                              'sample size': self.variant_sample_size,
                              'conversions': self.variant_conversions,
                              '95% credible interval': ('{:.5f}'.format(results['variant_cvr_lower']),
                                                        '{:.5f}'.format(results['variant_cvr_upper']))},
                  'outcome': {'uplift': '{:.3f}%'.format(uplift * 100),
                              'control expected loss': results['control_expected_loss'],
                              'variant expected loss': results['variant_expected_loss'],
                              'prob (variant >= control)': '{:.3f}%'.format(
                                  results['prob_variant_better_than_control'] * 100)
                              # ,'standard error': results['prob_variant_better_control_error']
                              }
                  }
        raw = pd.DataFrame(output).reset_index()
        raw_finished = self.transform_frame(raw)
        return raw_finished

    def plot_expected_loss(self,
                           df: pd.DataFrame,
                           epsilon=0.0001):
        """
        take posterior then calculate joint posterior
        return expected losses
        """
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        lw = 1
        alpha = 1

        filtered = df
        control = ax2.plot(filtered['day'],
                           filtered['control_expected_loss'],
                           label='choosing control',
                           linewidth=lw,
                           color='r',
                           alpha=alpha)
        treatment = ax2.plot(filtered['day'],
                             filtered['variant_expected_loss'],
                             linewidth=lw,
                             label='choosing variant',
                             color='blue',
                             alpha=alpha)

        ax2.axhline(epsilon,
                    color='black',
                    label='threshold={}'.format(epsilon),
                    ls='--')

        ax1.plot(filtered['day'],
                 filtered['control_mean'],
                 linewidth=1,
                 color='r',
                 label='control')
        ax1.fill_between(filtered['day'],
                         filtered['control_lower'],
                         filtered['control_upper'],
                         alpha=0.3,
                         color='r'
                         )
        ax1.plot(filtered['day'],
                 filtered['variant_mean'],
                 linewidth=1,
                 color='blue',
                 label='variant')
        ax1.fill_between(filtered['day'],
                         filtered['variant_lower'],
                         filtered['variant_upper'],
                         alpha=0.3,
                         color='blue'
                         )

        prob = ax3.plot(filtered['day'],
                        filtered['prob_variant_better_than_control'],
                        lw=lw,
                        color='purple',
                        label='P(λv>λc)',
                        alpha=alpha)
        ax3.set_ylim([0, 1])
        ax3.axhline(0.5,
                    color='black',
                    lw=lw,
                    ls='--')

        ax2.set_xlabel('Day')
        ax2.set_title('Expected Loss')
        ax2.legend()

        ax1.set_xlabel('Day')
        ax1.set_title('Cumulative performance')
        ax1.legend()

        ax3.set_xlabel('Day')
        ax3.set_title('Probability Variant better than Control')

        plt.show()

    ############################################################################################
    ############################################################################################
    ############################################################################################

    def beta(self,
             sample,
             prior):
        """
        Beta distribution - models the probability (p) of success.
        take samples A and B, then return posterior
        """
        posterior = np.random.beta(prior['alpha'] + sum(sample),
                                   prior['beta'] + len(sample) - sum(sample),
                                   size=len(sample))
        return posterior

    def poisson(self,
                sample,
                prior):
        """
        Poisson distribution - count model
        """
        posterior = np.random.gamma(shape=prior['shape'] + len(sample),
                                    scale=prior['rate'] + sum(sample),
                                    size=len(sample))
        return posterior

    def normal(self):
        pass

    def lognormal(self):
        pass

    def bayes_posterior(self,
                        sample_a,
                        sample_b,
                        prior,
                        distribution):
        """
        take sample of groups A and B, with corresponsive distribution function
            sample_a:
            sample_b:
            prior: list of prior parameters required for identifying distribution
            distribution: string - name of distribution, as of now, it can be, such as, 'beta', 'poisson', and 'normal'
        return posterior of groups A and B
        """
        distri_ = getattr(self, distribution)
        # if distribution.lower() == 'beta':
        posterior_a = distri_(sample_a,
                              prior)
        posterior_b = distri_(sample_b,
                              prior)
        return posterior_a, posterior_b

    def expected_loss_calculation(self,
                                  pos_a,
                                  pos_b):
        """
        take 2 posteriors then return expected losses of choosing each
        3-step process:
            1. get diff list between sample a and sample b
            2. for each item in each list, sum them up
            3. compute the mean of each list
        """
        b_better_than_a = [i <= j for i, j in zip(pos_a, pos_b)]
        winner_array = np.array(b_better_than_a)
        size_ = min(len(pos_a), len(pos_b))
        p = 1 / size_ * sum(winner_array)  # prob that b is better than a
        # p_error                                #error of calculating that prob
        loss_a = [max(j - i, 0) for i, j in zip(pos_a, pos_b)]
        loss_b = [max(i - j, 0) for i, j in zip(pos_a, pos_b)]

        expected_loss_a = np.sum([int(x) * y for x, y in zip(winner_array, loss_a)]) / (size_)
        expected_loss_b = np.sum([(1 - int(x)) * y for x, y in zip(winner_array, loss_b)]) / (size_)
        return expected_loss_a, expected_loss_b, p

    def main(self,
             sample_a,
             sample_b,
             prior,
             distribution):
        """
        """
        days = int(min(len(sample_a), len(sample_b)))
        records = []
        output = pd.DataFrame()
        for day in range(days):
            sequence_a = self.get_sequences_from_df(sample_a.iloc[:day + 1],
                                                    self.sample_column,
                                                    self.conversion_column)
            sequence_b = self.get_sequences_from_df(sample_b.iloc[:day + 1],
                                                    self.sample_column,
                                                    self.conversion_column)
            pos_a, pos_b = self.bayes_posterior(sequence_a,
                                                sequence_b,
                                                prior=prior,
                                                distribution=distribution)
            expected_loss_a, expected_loss_b, p = self.expected_loss_calculation(pos_a, pos_b)
            mean_a = np.mean(pos_a)
            a_lower, a_upper = self.calc_min_interval(pos_a, 0.05)
            mean_b = np.mean(pos_b)
            b_lower, b_upper = self.calc_min_interval(pos_b, 0.05)

            records.append({  # 'simulations': i,
                'day': day,
                'control_expected_loss': expected_loss_a,
                'variant_expected_loss': expected_loss_b,
                'prob_variant_better_than_control': p,
                # 'prob_variant_better_control_error': p_error,
                'control_mean': mean_a,
                'control_lower': a_lower,
                'control_upper': a_upper,
                'variant_mean': mean_b,
                'variant_lower': b_lower,
                'variant_upper': b_upper
            })
            simulation_results = pd.DataFrame.from_records(records)
        self.results = simulation_results
        return simulation_results
