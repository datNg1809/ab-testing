import numpy as np
import pandas as pd
#import pymc3 as pm
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
                 simulations= 1
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
                 load_new:bool=True,
                 save_to_disc:bool=True,
                 query=None):

        if not load_new:
            df = self.load_data_from_disc()
        else:
            df = db.import_sf_sql(query)
            if save_to_disc:
                file_name = self.make_folder()
                df.to_csv(file_name, sep=',', index=False, compression=None)
        return df

    def load_data_from_disc(self):
        file_name = self.make_folder()
        df = pd.read_csv(file_name, sep=',', index_col=False, compression=None)
        return df

    def make_folder(self):
        path = os.environ['FILE_PATH']
        file_path = os.path.join(path, self.project_name)
        os.makedirs(file_path, exist_ok=True)
        file_name = (str(file_path) + '/' + str(self.project_name) + '.csv')
        return file_name
    
    def prepare_data(self,
                     df:pd.DataFrame,
                     day_index:bool=True,
                     control_filter:str=None,
                     variant_filter:str=None,
                     date_column:str=None,
                     sample_column:str=None,
                     conversion_column:str=None,
                     test_column:str=None
                    ):
        
        if control_filter is None: control_filter = self.control_filter
        if variant_filter is None: variant_filter = self.variant_filter
        if date_column is None: date_column = self.date_column
        if sample_column is None: sample_column = self.sample_column
        if conversion_column is None: conversion_column = self.conversion_column
        if test_column is None: test_column = self.test_column
        
        df = df[[date_column, sample_column, test_column, conversion_column]]
        control_dataset = df[df[test_column]==control_filter].groupby(date_column).agg({sample_column:'count',
                                                                                        conversion_column:'sum'})
        variant_dataset = df[df[test_column]==variant_filter].groupby(date_column).agg({sample_column:'count',
                                                                                        conversion_column:'sum'})
        if day_index:
            c = self.make_day_index(control_dataset, date_column)
            v = self.make_day_index(variant_dataset, date_column)
        else:
            c = control_dataset
            v = variant_dataset
        
        #Override default values when declaring class
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

    def get_sequences_from_df(self, df:pd.DataFrame, n_column:str, k_column:str):
        def sequence(n:int, k:int):
            _n=int(n)
            _k=int(k)
            return np.concatenate([np.zeros(_n-_k),np.ones(_k)])
        
        sequences= np.concatenate(
            df.apply(lambda row: sequence(row[n_column],row[k_column]), axis=1).values)
        np.random.shuffle(sequences)

        return sequences
    
    # def get_CI(self,
    #            _array:np.array,
    #            alpha:int=0.95,
    #            mode:str='ci95'
    #           ):
    #     mean, var, std = stats.bayes_mvs(_array, alpha)
    #     if mode=='ci95':
    #         return mean.statistic, mean.minmax[0], mean.minmax[1]
    #     elif mode=='std':
    #         return std.statistic
    
    def posterior_analytic_importance_sampling(self,
                                               control:pd.DataFrame, 
                                               variant:pd.DataFrame,
                                               sample_col:str=None,
                                               conversion_col:str=None,
                                               days:int=None,
                                               prior_alpha=None,
                                               prior_beta=None,
                                               simulations=None
                                                ):
        """Uses Importance Sampling technique"""
        if prior_alpha == None:
            prior_alpha = self.alpha_prior
        if prior_beta == None:
            prior_beta = self.beta_prior
        
        if days == None:
            days = int(min(len(control),len(variant)) )
        elif days >= min(len(control),len(variant)):
            days = int(min(len(control),len(variant)) )
        
        if sample_col is None:
            sample_col = self.sample_column
        if conversion_col is None:
            conversion_col = self.conversion_column
        
        if simulations is None: simulations=self.simulations
        
        output = pd.DataFrame()
        N_mc = 500000   # Monte Carlo intergration - importance sampling technique
        
        for i in range(1,simulations+1):
            records = []
            for day in range(days):
                control_ = self.get_sequences_from_df(control.iloc[:day+1],sample_col,conversion_col)
                variant_ = self.get_sequences_from_df(variant.iloc[:day+1],sample_col,conversion_col)

                mean_c, var_c = stats.beta.stats(a=prior_alpha+sum(control_),
                                                 b=prior_beta+len(control_)-sum(control_),
                                                 moments='mv')
                mean_v, var_v = stats.beta.stats(a=prior_alpha+sum(variant_),
                                                 b=prior_beta+len(variant_)-sum(variant_),
                                                 moments='mv')
                # quantiles - area for pdf calculation
                randx_c = np.random.normal(loc=mean_c, 
                                           scale=1.25*np.sqrt(var_c), 
                                           size=N_mc)
                randx_v = np.random.normal(loc=mean_v, 
                                           scale=1.25*np.sqrt(var_v), 
                                           size=N_mc)
                # Posterior f_x
                f_c = stats.beta.pdf(randx_c,
                                     a = prior_alpha+sum(control_), 
                                     b = prior_beta+len(control_)-sum(control_))
                f_v = stats.beta.pdf(randx_v,
                                     a = prior_alpha+sum(variant_), 
                                     b = prior_beta+len(variant_)-sum(variant_))

                # Transformation I ~ 1/N * sum(f_x/g_x) while f_x ~ Beta distribution and g_x ~ normal distribution
                g_c = stats.norm.pdf(randx_c,
                                     loc=mean_c, 
                                     scale=1.25*np.sqrt(var_c))
                g_v = stats.norm.pdf(randx_v,
                                     loc=mean_v, 
                                     scale=1.25*np.sqrt(var_v))

                # Joint posterior
                y = (f_c * f_v)/(g_c * g_v)
                y_v = y[randx_v >= randx_c]  # variant is better than control
                
                p = 1/N_mc * sum(y_v)
                p_error = np.sqrt(1*(y_v*y_v).sum()/N_mc - (1*y_v.sum()/N_mc)**2)/np.sqrt(N_mc)

                c_lower, c_upper = stats.beta.interval(0.95,
                                                       a=prior_alpha+sum(control_),
                                                       b=prior_beta+len(control_)-sum(control_)
                                                      )
                v_lower, v_upper = stats.beta.interval(0.95,
                                                       a=prior_alpha+sum(variant_),
                                                       b=prior_beta+len(variant_)-sum(variant_)
                                                      )

                # Expected loss
                y_loss_control = ((randx_v-randx_c)*y)[randx_v >= randx_c]
                loss_control = 1/N_mc * sum(y_loss_control)

                y_loss_variant = ((randx_c-randx_v)*y)[randx_c >= randx_v]
                loss_variant = 1/N_mc * sum(y_loss_variant)

                records.append({'simulations': i,
                                'day': day,
                                'variant_expected_loss': loss_variant, 
                                'control_expected_loss': loss_control,
                                'prob_variant_better_than_control': p,
                                'prob_variant_better_control_error': p_error,
                                'control_cvr': mean_c,
                                'control_cvr_lower': c_lower,
                                'control_cvr_upper': c_upper,
                                'variant_cvr': mean_v,
                                'variant_cvr_lower': v_lower,
                                'variant_cvr_upper': v_upper
                               })
            simulation_results = pd.DataFrame.from_records(records)
            output = pd.concat([output, simulation_results])
            self.results = output
        return output

    def posterior_pymc(self, 
                       control:pd.DataFrame, 
                       variant:pd.DataFrame,
                       sample_col=None,
                       conversion_col=None,
                       alpha=None,
                       beta=None,
                       day=None,
                       plot=True
                      ):
        """Could be used for double-checking"""
        if sample_col is None:
            sample_col = self.sample_column
        if conversion_col is None:
            conversion_col = self.conversion_column
        if alpha is None: alpha = self.alpha_prior
        if beta is None: beta = self.beta_prior
        
        
        control = self.get_sequences_from_df(control.iloc[:day], sample_col, conversion_col)
        variant = self.get_sequences_from_df(variant.iloc[:day], sample_col, conversion_col)
        
        
        with pm.Model() as m:
            #prior
            mu_control = pm.Beta('mu_control', alpha=alpha, beta=beta)
            mu_variant = pm.Beta('mu_variant', alpha=alpha, beta=beta)

            #Define the deterministic delta function
            lift = pm.Deterministic('lift', mu_variant - mu_control)
            
            #Likelihood function
            obs_control = pm.Bernoulli('obs_control', p=mu_control, observed=control)
            obs_variant = pm.Bernoulli('obs_variant', p=mu_variant, observed=variant)

            trace = pm.sample(5000,
                              tune=4500,
                              cores=3,
                              chains=3,
                              return_inferencedata=True)

            #lift = trace['lift']
            #print("Probability Variant is WORSE than Control: %.3f" % \
            #      np.mean(lift < 0))

            #print("Probability Variant is BETTER than Control: %.3f" % \
            #      np.mean(lift > 0))

            if plot:
                az.plot_posterior(trace,
                                  hdi_prob=0.95,
                                  round_to=5,
                                  kind='hist'
                                 )
                
            #posterior = {'mu_control':mu_control,
            #             'mu_variant':mu_variant,
            #             'lift':lift}
        return trace
    
    def plot_lines(self, 
                   control:pd.DataFrame, 
                   variant:pd.DataFrame, 
                   conversion_column=None,
                   sample_column=None
                    ):
        if conversion_column is None:
            conversion_column=self.conversion_column
        if sample_column is None:
            sample_column=self.sample_column
        
        fig, ax = plt.subplots(1, 1, figsize=(10,6))
        ax.plot(control.index, control[conversion_column]/control[sample_column], label='control_cvr')
        ax.plot(variant.index, variant[conversion_column]/variant[sample_column], label='variant_cvr')
        ax.legend()
        ax.set_xlabel('Day')
        ax.set_title('CvR by Day')
        plt.show()

    def plot_expected_loss(self, 
                           df:pd.DataFrame, 
                           epsilon=0.0001):
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
                 filtered['control_cvr'], 
                 linewidth=1, 
                 color='r',
                 label='control')
        ax1.fill_between(filtered['day'],
                         filtered['control_cvr_lower'],
                         filtered['control_cvr_upper'],
                         alpha=0.3,
                         color='r'
                        )
        ax1.plot(filtered['day'], 
                 filtered['variant_cvr'], 
                 linewidth=1, 
                 color='blue',
                 label='variant')
        ax1.fill_between(filtered['day'],
                         filtered['variant_cvr_lower'],
                         filtered['variant_cvr_upper'],
                         alpha=0.3,
                         color='blue'
                        )

        prob = ax3.plot(filtered['day'],
                         filtered['prob_variant_better_than_control'],
                         lw=lw,
                         color = 'purple',
                         label = 'P(λv>λc)',
                         alpha=alpha)
        ax3.set_ylim([0,1])
        ax3.axhline(0.5,
                    color='black',
                    lw=lw,
                    ls='--')

        ax2.set_xlabel('Day')
        ax2.set_title('Expected Loss')
        ax2.legend()

        ax1.set_xlabel('Day')
        ax1.set_title('Conversion Rates')
        ax1.legend()

        ax3.set_xlabel('Day')
        ax3.set_title('Probability Variant better than Control')          

        plt.show()

    def get_results(self):
        results = self.results
        results = results.iloc[-1]

        uplift = (results['variant_cvr'] - results['control_cvr'])/results['control_cvr']

        output = {'control': {'CvR': '{:.5f}'.format(results['control_cvr']),
                              'sample size': self.control_sample_size,
                              'conversions': self.control_conversions,
                              '95% credible interval': ('{:.5f}'.format(results['control_cvr_lower']),'{:.5f}'.format(results['control_cvr_upper']))},
                  'variant': {'CvR': '{:.5f}'.format(results['variant_cvr']),
                              'sample size': self.variant_sample_size,
                              'conversions': self.variant_conversions,
                              '95% credible interval': ('{:.5f}'.format(results['variant_cvr_lower']),'{:.5f}'.format(results['variant_cvr_upper']))},
                  'outcome': {'uplift': '{:.3f}%'.format(uplift*100),
                              'control expected loss': results['control_expected_loss'],
                              'variant expected loss': results['variant_expected_loss'],
                              'prob (variant >= control)': '{:.3f}%'.format(results['prob_variant_better_than_control']*100),
                              'standard error': results['prob_variant_better_control_error']}}

        raw = pd.DataFrame(output).reset_index()
        raw_finished = self.transform_frame(raw)
        return raw_finished

    def transform_frame(self, df):
        #metric_order = ['uplift', 'control expected loss', 'variant expected loss', 'prob (variant >= control)', 'standard error']
        df = df.melt(id_vars='index', value_name='VALUE').set_index(['variable', 'index']).dropna()
        df.index.names = ['GROUP', 'METRIC']

        #new = df.reindex(level='metric', labels=metric_order)
        #df = df.loc[['control', 'variant']].append(new)
        return df



        
        
    # def posterior_analytic(self,
    #                        control:pd.DataFrame,
    #                        treatment:pd.DataFrame,
    #                        sample_col:str=None,
    #                        conversion_col:str=None,
    #                        days:int=None,
    #                        prior_alpha=None,
    #                        prior_beta=None,
    #                        simulations:int=None
    #                         ):
    #     """Uses crude sampling technique"""
    #     if prior_alpha == None:
    #         prior_alpha = self.alpha_prior
    #     if prior_beta == None:
    #         prior_beta = self.beta_prior
    #
    #     if (days >= min(len(control),len(treatment))) or (days == None):
    #         days = int(min(len(control),len(treatment)) - 1)
    #
    #     if simulations == None:
    #         simulations = 30
    #
    #     if sample_col is None:
    #         sample_col = self.sample_column
    #     if conversion_col is None:
    #         conversion_col = self.conversion_column
    #
    #     output = pd.DataFrame()
    #     simulation = 0
    #
    #     for i in range(simulations):
    #         simulation += 1
    #         records = []
    #         for day in range(days):
    #             control_simulation = self.get_sequences_from_df(control.iloc[:day+1],
    #                                                             sample_col,conversion_col)
    #             treatment_simulation = self.get_sequences_from_df(treatment.iloc[:day+1],
    #                                                               sample_col,conversion_col)
    #             size = min(len(control_simulation),len(treatment_simulation))
    #             control_conversions = control_simulation.sum()
    #             treatment_conversions = treatment_simulation.sum()
    #
    #             # Posterior sampling
    #             control_pdfs = np.random.beta(prior_alpha + control_conversions,
    #                                           prior_beta + len(control_simulation) - control_conversions,
    #                                           size=size)
    #             treatment_pdfs = np.random.beta(prior_alpha + treatment_conversions,
    #                                             prior_beta + len(treatment_simulation) - treatment_conversions,
    #                                             size=size)
    #             treatment_pdf_higher = [i <= j for i,j in zip(control_pdfs, treatment_pdfs)]
    #             treatment_win_array = np.array(treatment_pdf_higher)
    #             p = 1/size * sum(treatment_win_array)
    #             p_error = np.sqrt(1*(treatment_win_array*treatment_win_array).sum()/size - (1*treatment_win_array.sum()/size)**2)/np.sqrt(size)
    #             expected_loss_control, expected_loss_treatment = self.calculate_expected_loss(control_pdfs,
    #                                                                                          treatment_pdfs,
    #                                                                                         treatment_pdf_higher)
    #             records.append({'simulation':simulation,
    #                             'day': day,
    #                             #'treatment_cr': (treatment_conversions/len(treatment_simulation)),
    #                             #'control_cr': (control_conversions/len(control_simulation)),
    #                             'treatment_expected_loss': expected_loss_treatment,
    #                             'control_expected_loss': expected_loss_control,
    #                             'prob_variant_better_than_control':np.sum(treatment_pdf_higher)/size,
    #                             'prob_variant_better_than_control_error': p_error
    #                             })
    #             simulation_results = pd.DataFrame.from_records(records)
    #         output = pd.concat([output, simulation_results])
    #     output = output.groupby('day').mean()
    #
    #     return output.drop('simulation',axis=1).reset_index()

    # def calculate_expected_loss(self,
    #                             control_simulation,
    #                             treatment_simulation,
    #                             treatment_won,
    #                             min_difference_delta=0):
    #     loss_control = [max((j - min_difference_delta) - i, 0) for i,j in zip(control_simulation,
    #                                                                           treatment_simulation)]
    #     loss_treatment = [max(i - (j - min_difference_delta), 0) for i,j in zip(control_simulation,
    #                                                                             treatment_simulation)]
    #     all_loss_control = [int(i)*j for i,j in zip(treatment_won, loss_control)]
    #     all_loss_treatment = [(1 - int(i))*j for i,j in zip(treatment_won, loss_treatment)]
    #     expected_loss_control = np.mean(all_loss_control)
    #     expected_loss_treatment = np.mean(all_loss_treatment)
    #
    #     return expected_loss_control, expected_loss_treatment
        

    # def generate_sample(self,
    #                     min_visits=None,
    #                     max_visits=None,
    #                     probability_control=None,
    #                     probability_variant=None,
    #                     days=None
    #                     ):
    #     """This is for testing purpose"""
    #
    #     day_number = np.array(range(1,days+1))
    #
    #     control_group = np.array(["control" for x in range(days)])
    #     control_sessions = np.random.randint(low=min_visits, high=max_visits, size=(days,))
    #     control_cvr = np.random.uniform(probability_control*0.8, probability_control*1.2, days)
    #     control_leads = control_sessions * control_cvr
    #     c = pd.DataFrame(data={'day_number':day_number,
    #                            'group':control_group,
    #                            'sessions':control_sessions,
    #                            'leads':control_leads})
    #     c = c.round()
    #
    #     variant_group = np.array(["test" for x in range(days)])
    #     variant_sessions = control_sessions
    #     variant_cvr = np.random.uniform(probability_variant*0.8, probability_variant*1.2, days)
    #     variant_leads = variant_sessions * variant_cvr
    #     v = pd.DataFrame(data={'day_number':day_number,
    #                            'group':variant_group,
    #                            'sessions':variant_sessions,
    #                            'leads':variant_leads})
    #     v = v.round()
    #
    #     df = pd.concat([c, v])
    #     return c,v
    #
    #
    # def run_multiple_simulations(self,
    #                              min_visits=None,
    #                              max_visits=None,
    #                              probability_control=None,
    #                              probability_variant=None,
    #                              days=None,
    #                              simulations=50,
    #                              plot:bool=True):
    #     """This is for testing purpose"""
    #
    #     dataset = pd.DataFrame()
    #     #columns=['simulation','day','group','sessions','leads']
    #
    #     for i in range(1,simulations+1):
    #         c,v = self.generate_sample(min_visits,
    #                                     max_visits,
    #                                     probability_control,
    #                                     probability_variant,
    #                                     days)
    #
    #         si = self.posterior_analytic_importance_sampling(
    #                                                        control=c,
    #                                                        treatment=v)
    #
    #         si['simulations']=i
    #         dataset=pd.concat([dataset,si])
    #
    #     #plotting
    #     if plot:
    #         self.plot_expected_loss(dataset,simulations=simulations)
    #     return dataset
    #
    #
    def calc_min_interval(self, x, alpha):
        """Internal method to determine the minimum interval of a given width
        Assumes that x is sorted numpy array.
        """

        n = len(x)
        cred_mass = 1.0-alpha

        interval_idx_inc = int(np.floor(cred_mass*n))
        n_intervals = n - interval_idx_inc
        interval_width = x[interval_idx_inc:] - x[:n_intervals]

        if len(interval_width) == 0:
            raise ValueError('Too few elements for interval calculation')

        min_idx = np.argmin(interval_width)
        hdi_min = x[min_idx]
        hdi_max = x[min_idx+interval_idx_inc]
        return hdi_min, hdi_max

    def hpd(self, x, alpha=0.05):
        """Calculate highest posterior density (HPD) of array for given alpha.
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
            tx = np.transpose(x, list(range(x.ndim))[1:]+[0])
            dims = np.shape(tx)
            # Container list for intervals
            intervals = np.resize(0.0, dims[:-1]+(2,))

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
        
        
    