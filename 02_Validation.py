# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
# ---

# %% [markdown]
# # Validation

# %% [markdown]
# A key step for evaluating any cognitive model is to find and measure model fit to real behavioral datasets.
# This process requires a few new sections of code. These include:
#
# 1. A `prepare_data` function that loads data from a provided path and formats it to support both efficient
#    model fitting and data visualization.
# 2. A `likelihood` function that computes the likelihood of the data given a model and a specified parameter
#    configuration.
# 3. A `likelihood_search` function that searches for the parameter configuration of a specified model that
#    makes the provided data the most likely.
# 4. A `visualize_fit` function that visually compares a selected dataset with a model fitted to it using
#    figures such as the serial position and conditional recall probability curve.
# 5. We may also need to try optimizing `models.InstanceCMR` to help speed up fitting, which can be a
#    pretty time-consuming process!
#
# We will use this code to fit our updated model to data sourced for a 2018 study focusing on organizational dynamics  
# in free recall of stories, cited below. To confirm that our discovered parameter configuration
# really works, we'll visualize the resulting fit. We'll also add some tests confirming the robustness of the fitting 
# algorithm.

# %% [markdown]
# > Cutler, R., Palan, J., Brown-Schmidt, S., & Polyn, S. (2019). Semantic and temporal structure in memory for narratives: A benefit for semantically congruent ideas. Context and Episodic Memory Syposium. 
# 
# # %% [markdown]
# ## Data Preparation

# %% [markdown]
# To support easy visalization and model fitting, we need a few formats for our data. First, we need a hierarchical textual representation of each story, dividing it into sentences and human-coded idea units for encoding into our model. Second, we need an array representation of the recall order of each unit, encoding for each recall position in each trial the index of each idea unit recalled. Third, to enable visualization of model fit, we need the Psifr format that the data already comes in. 
# 
# This Psifr format is annotated with the text of the idea units each item corresponds to. Those will define our units for the sake of validation. To map these to corresponding cycles, we will use a punctuation heuristic, grouping units based on the occurence of periods. Finer tokenization would be possible with more time.

# %%
#export
import scipy.io as sio
import numpy as np
import pandas as pd
from psifr import fr

def prepare_brownschmidt_data(path, story):
    """
    Prepares data formatted in a text-annotated Psifr format for model fitting. 

    Loads data from `path` with appropriate format and returns a selected dataset as an array of
    unique recall trials and a dataframe of unique study and recall events organized according to `psifr`
    specifications.

    **Arguments**:  
    - path: source of data file  
    - dataset_index: index of the dataset to be extracted from the file

    **Returns**:
    - trials: int64-array where rows identify a unique trial of responses and columns corresponds to a unique recall
        index.  
    - merged: as a long format table where each row describes one study or recall event.  
    - list_length: length of lists studied in the considered dataset
    """

    # load dataset data from psifr format
    df = pd.read_csv(path, sep='\t')
    df = df.loc[df['story']==story]

    # build units_and_cycles representation for each story
    units = df['text'].unique()
    units_and_cycles = [[units[0]]]
    for unit in units[1:]:
        if unit[0].isupper():
            units_and_cycles.append([])
        units_and_cycles[-1].append(unit)

    # encode recall events into trial array
    recalls = df.loc[df['trial_type'] == 'recall']
    trials = []
    last_trial = 0
    for index, row in recalls.iterrows():

        # start new vector if trial changes
        if row['trial'] != last_trial:
            trials.append([])
        trials[-1].append(row['item'])

        last_trial = row['trial']

    list_length = np.max(df['position'].unique())
    return trials, df, units_and_cycles, list_length

# %% [markdown]
# We can generate a quick preview of some datasets using this function.

# %%
text_trials, text_events, text_units_and_cycles, text_length = prepare_brownschmidt_data('data/sequences/human/clean_human.csv', 'Fisherman')

text_events.head()

# %% [markdown]
# ## Configuring the Parameter Search
# To fit the model to some dataset, we must specify a cost function that scales against the likelihood that the model
# with a specified parameter configuration could have generated the specified dataset.

# %%
#export
#hide
import numpy as np
from landscape import Landscape
from numba import njit

#@njit(fastmath=True, nogil=True)
def data_likelihood(trials, sbert_model_name,
                 initial_max_activation=1.0,
                 initial_decay_rate=0.1,
                 initial_memory_capacity=5.0,
                 initial_learning_rate=0.9,
                 initial_semantic_strength_coeff=1.0):
    """
    Generalized cost function for fitting the Landscape model optimized using the numba library.
    
    Output scales inversely with the likelihood that the model and specified parameters would generate the specified
    trials. For model fitting, is usually wrapped in another function that fixes and frees parameters for optimization.

    **Arguments**:
    - trials: int64-array where rows identify a unique trial of responses and columns corresponds to a unique recall
      index.
    - A configuration for each parameter of model

    **Returns** the negative sum of log-likelihoods across specified trials conditional on the specified parameters and
    the mechanisms of InstanceCMR.
    """

    model = Landscape(sbert_model_name, initial_max_activation, initial_decay_rate, initial_memory_capacity, 
        initial_learning_rate, initial_semantic_strength_coeff)
    
    model.experience(np.eye(item_count, item_count + 1, 1))
    
    likelihood = np.ones(np.shape(trials))
    
    for trial_index in range(len(trials)):
        trial = trials[trial_index]
        
        for recall_index in range(len(trial)):
            recall = trial[recall_index]
            likelihood[trial_index, recall_index] = \
                model.outcome_probabilities()[recall]
            model.force_recall(recall)
            if recall == 0:
                break
        
    return -np.sum(np.log(likelihood))

# %% [markdown]
# We return a sum of negative log likelihoods because our fitting functions search for parameters that minimize
# error functions and log-likelihoods are negative. For example,

# %%

lb = np.finfo(float).eps
hand_fit_parameters = {
    'item_count': murd_length,
    'encoding_drift_rate': .8,
    'start_drift_rate': .7,
    'recall_drift_rate': .8,
    'shared_support': 0.01,
    'item_support': 1.0,
    'learning_rate': .3,
    'primacy_scale': 1,
    'primacy_decay': 1,
    'stop_probability_scale': 0.01,
    'stop_probability_growth': 0.3,
    'choice_sensitivity': 2
}
data_likelihood(text_trials[:60], **hand_fit_parameters)

# %% [markdown]
# returns `1154.6050515722645`.

# %% [markdown]
# When it comes to model fitting, we will normally wrap this loss function within another that fixes function
# parameters we aren't seeking a fit for - for example, our data structure `trials`. In general, our true
# objective function should fill free parameters of `data_likelihood` using values in a single `x` array.

# %%
# export
# hide

def generate_objective_function(data_to_fit, fixed_parameters, free_parameters):
    """
    Generates and returns an objective function for input to support search through parameter space for a model
    fit using an optimization function.

    Required model_class attributes:  
    - cycles: adding a new experience to the memory model  
    - force_recall: forces recall of item, ignoring model state  
    - outcome_probabilities: returns item supports given activations

    Other arguments:  
    - fixed_parameters: dictionary mapping parameter names to values they'll be fixed to during search,
        overloaded by free_parameters if overlap
    - free_parameters: list of strings naming parameters for fit during search
    - data_to_fit: array where rows identify a unique trial of responses and columns corresponds to a unique
        recall index

    Returns a function that accepts a vector x specifying arbitrary values for free parameters and returns
    evaluation of data_likelihood using the model class, all parameters, and provided data.
    """
    
    return lambda x: data_likelihood(data_to_fit, **{**fixed_parameters, **{
        free_parameters[i]:x[i] for i in range(len(x))}})


# %%
try:
    show_doc(generate_objective_function, title_level=3)
except:
    pass

# %% [markdown]
# We can generate and apply the function to compute loss for different configurations of just
# `encoding_drift_rate` with code like the following:

# %%
cost_function = generate_objective_function(text_trials[:60], hand_fit_parameters, ['encoding_drift_rate'], )

cost_function([.8]), cost_function([.3])

# %% [markdown]
# Which returns (again) `1154.6050515722645`, and `1338.7420002341266`.

# %% [markdown]
# From here, the function searching the parameter space to find model configurations that fit as well to data
# as possible is mostly written for us. We use the `differential_evolution` function from `scipy.optimize`. The
# entire function specification can be found in [the corresponding
# docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html), but
# its main requirements are a cost function to be minimized and a `bounds` array specifying in each row a (min,
# max) pair constraining search over each parameter. With the cost function generated above, we can find the
# best fitting value of `encoding_drift_rate` to `murd_trials[:60]` for our `InstanceCMR` class very
# efficiently after specifying a bounds array that constrains search between 0 and 1:

# %%
from scipy.optimize import differential_evolution

result = differential_evolution(cost_function, [(np.finfo(float).eps, 1-np.finfo(float).eps)], disp=True)
result

# %% [markdown]
# This function returns an output with the following attributes:
#
# ```
#      fun: 1153.963638767822
#      jac: array([0.])
#  message: 'Optimization terminated successfully.'
#     nfev: 57
#      nit: 2
#  success: True
#        x: array([0.81981498])
# ```
#
# The `x` attribute of the result object contains the best parameter configuration found, while the `fun`
# attribute represents the overall cost of the configuration as computed with our specified cost function. 

# %% [markdown]
# ## Results
# We can visually compare the behavior of the model with these parameters against the data it's fitted to with a new
# `visualize_fit` function.

# %%
# export
# hide
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_fit(model_class, parameters, data, data_query=None, experiment_count=1000, savefig=False):
    """
    Apply organizational analyses to visually compare the behavior of the model with these parameters against
    specified dataset.
    """
    
    # generate simulation data from model
    model = model_class(**parameters)
    try:
        model.experience(np.eye(model.item_count, model.item_count + 1, 1))
    except ValueError:
        model.experience(np.eye(model.item_count, model.item_count))
    sim = []
    for experiment in range(experiment_count):
        sim += [[experiment, 0, 'study', i + 1, i] for i in range(model.item_count)]
    for experiment in range(experiment_count):
        sim += [[experiment, 0, 'recall', i + 1, o] for i, o in enumerate(model.free_recall())]
    sim = pd.DataFrame(sim, columns=['subject', 'list', 'trial_type', 'position', 'item'])
    sim_data = fr.merge_free_recall(sim)
    
    # generate simulation-based spc, pnr, lag_crp
    sim_spc = fr.spc(sim_data).reset_index()
    sim_pfr = fr.pnr(sim_data).query('output <= 1') .reset_index()
    sim_lag_crp = fr.lag_crp(sim_data).reset_index()
    
    # generate data-based spc, pnr, lag_crp
    data_spc = fr.spc(data).query(data_query).reset_index()
    data_pfr = fr.pnr(data).query('output <= 1').query(data_query).reset_index()
    data_lag_crp = fr.lag_crp(data).query(data_query).reset_index()
    
    # combine representations
    data_spc['Source'] = 'Data'
    sim_spc['Source'] = model_class.__name__
    combined_spc = pd.concat([data_spc, sim_spc], axis=0)
    
    data_pfr['Source'] = 'Data'
    sim_pfr['Source'] = model_class.__name__
    combined_pfr = pd.concat([data_pfr, sim_pfr], axis=0)
    
    data_lag_crp['Source'] = 'Data'
    sim_lag_crp['Source'] = model_class.__name__
    combined_lag_crp = pd.concat([data_lag_crp, sim_lag_crp], axis=0)
    
    # generate plots of result
    # spc
    g = sns.FacetGrid(dropna=False, data=combined_spc)
    g.map_dataframe(sns.lineplot, x='input', y='recall', hue='Source')
    g.set_xlabels('Serial position')
    g.set_ylabels('Recall probability')
    plt.title('P(Recall) by Serial Position Curve')
    g.add_legend()
    g.set(ylim=(0, 1))
    if savefig:
        plt.savefig('figures/{}_fit_spc.jpeg'.format(model_class.__name__), bbox_inches='tight')
    else:
        plt.show()
    
    #pdf
    h = sns.FacetGrid(dropna=False, data=combined_pfr)
    h.map_dataframe(sns.lineplot, x='input', y='prob', hue='Source')
    h.set_xlabels('Serial position')
    h.set_ylabels('Probability of First Recall')
    plt.title('P(First Recall) by Serial Position')
    h.add_legend()
    h.set(ylim=(0, 1))
    if savefig:
        plt.savefig('figures/{}_fit_pfr.jpeg'.format(model_class.__name__), bbox_inches='tight')
    else:
        plt.show()
    
    # lag crp
    max_lag = 5
    filt_neg = f'{-max_lag} <= lag < 0'
    filt_pos = f'0 < lag <= {max_lag}'
    i = sns.FacetGrid(dropna=False, data=combined_lag_crp)
    i.map_dataframe(
        lambda data, **kws: sns.lineplot(data=data.query(filt_neg),
                                         x='lag', y='prob', hue='Source', **kws))
    i.map_dataframe(
        lambda data, **kws: sns.lineplot(data=data.query(filt_pos),
                                         x='lag', y='prob', hue='Source', **kws))
    i.set_xlabels('Lag')
    i.set_ylabels('Recall Probability')
    plt.title('Recall Probability by Item Lag')
    i.add_legend()
    i.set(ylim=(0, 1))
    if savefig:
        plt.savefig('figures/{}_fit_crp.jpeg'.format(model_class.__name__), bbox_inches='tight')
    else:
        plt.show()


# %%
try:
    show_doc(visualize_fit, title_level=3)
except:
    pass

# %% [markdown]
# With the function we can plot the model with the resulting parameter configuration against the actual data in
# one line:

# %%
visualize_fit(Landscape, {**hand_fit_parameters, **{'encoding_drift_rate': result.x[0]}}, murd_events, 'subject == 1', experiment_count=1000, savefig=True)