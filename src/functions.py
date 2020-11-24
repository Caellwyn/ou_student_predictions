import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder


def load_OU_data(prediction_window = None):
    """
    Loads data from '/content' folder, and prepares it for modeling.  It imports only data up to the relative date from course start that is passed as the argument 'prediction_window'.
    concatenates data from:
    * studentRegistration.csv
    * courses.csv
    * studentInfo.csv
    * studentVle.csv
    * vle.csv
    * studentAssessment.csv
    * assessments.csv
    
    Then: 
    * drops rows with missing and suspicious data, 
    * derives features: number of days studied, number of activities engaged, and number of clicks for each 
    registration over the course of the module.  Also derives assessment score average and number of 
    assessments completed within the prediction window.
    returns a new dataframe.
    Note, derived features should be scaled with CourseScaler()
    * 
    """
    import pandas as pd
    import numpy as np
    import zipfile

    # import data
    zf = zipfile.ZipFile('../content/anonymisedData.zip') 

    registrations = pd.read_csv(zf.open('studentRegistration.csv'))
    courseInfo = pd.read_csv(zf.open('courses.csv'))
    students = pd.read_csv(zf.open('studentInfo.csv'))
    student_vle = pd.read_csv(zf.open('studentVle.csv'))
    vle_info = pd.read_csv(zf.open('vle.csv'))
    student_assessments = pd.read_csv(zf.open('studentAssessment.csv'), skiprows=[128223,64073])
    assessments_info = pd.read_csv(zf.open('assessments.csv'))

    index_columns = ['code_module','code_presentation','id_student']
    
    # Registrations
    full_registrations = pd.merge(students, registrations, on=index_columns, validate='1:1')
    full_registrations = pd.merge(full_registrations, courseInfo, \
                             on=['code_module','code_presentation'], validate='many_to_one')
    full_registrations.dropna(subset=['date_registration','imd_band'], inplace=True)

    not_withdrawn = full_registrations['date_unregistration'].isna()
    
    if type(prediction_window) == float:
        prediction_window = prediction_window*(full_registrations.module_presentation_length.min())
    if prediction_window:
        withdrawn_after_predict = (full_registrations['final_result'] == 'Withdrawn') \
                                    & (full_registrations['date_unregistration'] > prediction_window)
    else:
        withdrawn_after_predict = (full_registrations['final_result'] == 'Withdrawn') \
                                    & (full_registrations['date_unregistration'] > 0)
    full_registrations = full_registrations[not_withdrawn | withdrawn_after_predict]
    full_registrations['date_unregistration'].fillna( \
        full_registrations['module_presentation_length'], inplace = True)
    full_registrations = full_registrations[full_registrations['date_unregistration']
                                            <= full_registrations['module_presentation_length']]
    if prediction_window:
        full_registrations.drop(columns=['date_unregistration'], inplace=True)

    # VLE
    if prediction_window:
        student_vle = student_vle[student_vle.date <= prediction_window]
    vle = pd.merge(student_vle,vle_info, 
                   how = 'left', \
                   on =['id_site','code_module', 'code_presentation'], \
                   validate = 'm:1').drop(columns = ['week_from','week_to'])

    total_activities = vle.groupby(by=index_columns).count().reset_index()
    total_activities = total_activities.drop(columns=['date','sum_click','activity_type'])

    date_grouped = vle.groupby(by=index_columns + ['date']).count().reset_index()
    days_studied = date_grouped.groupby(by=index_columns).count().reset_index()
    days_studied = days_studied.drop(columns=['id_site','sum_click','activity_type'])

    clicks = vle.groupby(by=index_columns).sum().reset_index()
    clicks = clicks.drop(columns=['id_site','date'])

    full_registrations = pd.merge(full_registrations, days_studied, on=index_columns)
    full_registrations = pd.merge(full_registrations, total_activities, on=index_columns)
    full_registrations = pd.merge(full_registrations, clicks, on=index_columns)

    # Assessments
    assessments = pd.merge(student_assessments, assessments_info, how='left', on='id_assessment')
    assessments.dropna(subset = ['score'], inplace=True)
    assessments.drop(columns = ['is_banked', 'weight'], inplace=True)
    if prediction_window:
        assessments = assessments[assessments['date_submitted'] <= prediction_window]
    num_assessments = assessments.groupby(by = index_columns).count().reset_index()
    num_assessments.drop(columns = ['date_submitted','score','date','assessment_type'], inplace=True)

    avg_score = assessments.groupby(by = index_columns).mean().reset_index()
    avg_score.drop(columns = ['date_submitted','id_assessment','date'], inplace=True)

    full_registrations = pd.merge(full_registrations, num_assessments, on=index_columns)
    full_registrations = pd.merge(full_registrations, avg_score, on=index_columns)
    
    # Rename columns
    new_cols = {'id_assessment':'assessments_completed',
                'score':'average_assessment_score','date':'days_studied',
                'id_site':'activities_engaged','sum_click':'total_clicks'}
    full_registrations = full_registrations.rename(columns = new_cols)

    return full_registrations



    
class CourseScaler(TransformerMixin, BaseEstimator):
    """
    ---
    Standardizes 'days_studied','activities_completed','total_clicks','assessments_completed', and 'average_score' if they are present in the dataframe for each course in the column 'code_module'.
    compatible with pipeline objects requiring fit and transform methods.
    ---
    methods:
    .fit(X) fits CourseScaler instance to dataframe
    .transform(X) transforms X if fit.
    .fit_transform(X)
    """
    def __init__(self, drop_course=True):
        self.drop_course = drop_course
        pass
    
    def fit(self,X,y=None):
        """
        fit(self,X,y=None)
        ---
        fits tranformer to a dataset, setting the means and standard deviations for each module.  Returns self
        ---
        X: the data to fit on.
        """
        import pandas as pd
        self.cols = ['days_studied','activities_engaged','total_clicks',\
                     'assessments_completed','average_assessment_score','num_of_prev_attempts']
        if len(self.cols) == 0:
            print('No columns to standardize')
            return self
        else:
            modules = X['code_module'].unique()
            self.means = pd.DataFrame(index = modules, columns = self.cols)
            self.stds = pd.DataFrame(index = modules, columns = self.cols)
            for module in modules:
                course_X = X[X['code_module'] == module]
                for col in self.cols:
                    mean = course_X[col].mean()
                    std = course_X[col].std()
                    self.means.loc[module, col] = mean
                    self.stds.loc[module, col] = std
            return self
        
    def transform(self,X,y=None):
        """
        transform(self,X,y=None)
        ---
        transforms X to be scaled according to fitted means and standard deviations for each module in 
        'code_module' by subtracting the mean and dividing by the standard deviation for the module the
        observation is in.
        Returns transformed data.
        ---
        X: data to be transformed.
        """
        import pandas as pd
        if not hasattr(self,'means'):
            print('WARNING: transformer Not yet fit')
            return self
        else:
            i = X.index
            X.reset_index(drop = True, inplace = True)
            scaled_X = pd.DataFrame()
            for module in self.means.index:
                course_X = X[X['code_module'] == module]
                for col in self.means.columns:
                    mean = self.means.loc[module,col]
                    std = self.stds.loc[module,col]
                    course_X[col] = (course_X[col] - mean)/std
                scaled_X = pd.concat([scaled_X,course_X], axis = 0)
            scaled_X.sort_index(inplace = True)
            scaled_X.index = i
            scaled_X.fillna(value = 0, inplace = True)
            if self.drop_course:
                scaled_X.drop('code_module', axis = 1, inplace = True)
            return scaled_X


def smotecourses(X,y,drop_course=True):
    """
    smotecourses(X,y,drop_course=True)
    ---
    This function goes through the data course by course and smotes each course individually using imblearn.over_sampling.SMOTE.  Because other transformers need the code_module variable to function, but it should not be given to a model, it has an optional flag to drop the code_module column.
    ---
    X: features in a pandas.DataFrame
    y: labels in an array-like structure
    drop_course: (optional) drops the code_module column.  Default is True.
    """
    from imblearn.over_sampling import SMOTE
    import pandas as pd
    
    X['label'] = y
    smoted_X = pd.DataFrame()
    smoted_y = pd.DataFrame()
    for module in X['code_module'].unique():
        course_df = X[X['code_module'] == module]
        if len(course_df) > 0:
            course_X = course_df.drop(columns = ['label','code_module'])
            course_y = course_df['label']
            course_X, course_y = SMOTE(random_state=111).fit_resample(course_X,course_y)
            if not drop_course:
                course_X['code_module'] = module
            smoted_X = pd.concat([smoted_X, course_X], axis=0)
            smoted_y = pd.concat([smoted_y, course_y], axis=0)
    smoted_X.fillna(value = 0, inplace = True)
    return smoted_X, smoted_y

def process_courses(X_train, y_train, X_test, y_test=None):
    """
    process_courses(X_train, y_train, X_test, y_test=None)
    ---
    Takes in training X and y and testing X (and testing y if you want, but it doesn't do anything to it).
    Returns X_train scaled by course and smoted by course, y_train smoted by course, X_test scaled using the mean and standard deviation of X_train, and y_test unchanged if you happen to pass it in by habit.  It happens.
    ---
    X_train: pandas.DataFrame of training variables
    y_train: array-like of training labels
    X_test: pandas.DataFrame of testing variables
    y_test: (optional) whatever, doesn't matter, just here so it doesn't break if you pass it my mistake.
    """
    cs = CourseScaler(drop_course=False)
    X_train = cs.fit_transform(X_train)
    X_test = cs.transform(X_test)
    X_train, y_train = smotecourses(X_train, y_train, drop_course = True)
    X_test = X_test.drop(columns = ['code_module'])
    if type(y_test) == type(None):
        return X_train, y_train, X_test
    else: 
        return X_train, y_train, X_test, y_test

def course_cross_validate(estimator,X,y,scoring='accuracy', cv=5, random_state = 111):
    """
    course_cross_validate(estimator,X,y,scoring='accuracy', cv=5, random_state = 111)
    ---
    Takes an estimator, features, X and targets y.  Splits X into cv number of folds and applies
    course-wise scaling to the train and test sets for each fold, 
    then applies course-wise smoting to the training set.  Fits each training fold, predicts
    each test fold, and returns the scores.
    ---
    estimator: an estimator object
    X: features in dataframe format
    y: labels as array or pandas series
    scoring: either 'accuracy' or 'f1'
    cv: number of folds
    random_state: random_state of Kfold generator
    """
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    import numpy
    if type(cv) == int:
        kf = KFold(n_splits = cv, shuffle=True, random_state=random_state)
    else: 
        kf = cv
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        X_train, y_train, X_test = process_courses(X_train, y_train, X_test)
        estimator.fit(X_train,y_train)
        y_pred = estimator.predict(X_test)
        if scoring == 'accuracy':
            from sklearn.metrics import accuracy_score
            scores.append(accuracy_score(y_test,y_pred))
        elif scoring == 'f1':
            from sklearn.metrics import f1_score
            scores.append(f1_score(y_test,y_pred))
    return scores
            
class Course_GridSearchCV():
    """
    Course_GridSearchCV(self, estimator, params, cv=5, scoring='accuracy', verbose = False)
    ---
    Grid search object using cross validation to find the best combination of parameters and keeping the best estimator and it's best cv scores and the mean of that list of cv scores.  It uses course_cross_validate to to cross validate each model using a custom and static pipeline to appropriately scale and smote each fold by course.
    ---
    estimator: an estimator with .fit(X,y) and .predict(X) methods.
    params: a dictionary of estimator hyperparameters of the format {'parameter_name':string or list of parameter values}
    cv: (optional) an int or cross validation generator object (like KFold), default = 5
    scoring: (optional) either 'accuracy' or 'f1_score'. Default = 'accuracy'
    verbose: (optional) if True, prints each attempted combination of parameter and their cross validation scores.
    """
    def __init__(self, estimator, params, cv=5, scoring='accuracy', verbose = False):
        """
        Course_GridSearchCV(self, estimator, params, cv=5, scoring='accuracy', verbose = False)
        ---
        Grid search object using cross validation to find the best combination of parameters and keeping the best estimator and it's best cv scores and the mean of that list of cv scores.  It uses course_cross_validate to to cross validate each model using a custom and static pipeline to appropriately scale and smote each fold by course.
        ---
        estimator: an estimator with .fit(X,y) and .predict(X) methods.
        params: a dictionary of estimator hyperparameters of the format {'parameter_name':string or list of parameter values}
        cv: (optional) an int or cross validation generator object (like KFold), default = 5
        scoring: (optional) either 'accuracy' or 'f1_score'. Default = 'accuracy'
        verbose: (optional) if True, prints each attempted combination of parameter and their cross validation scores.
        Attributes:
           .estimator: base estimator
           .paramgrid: an sklearn.model_selection.ParameterGrid object enumerating all combinations of the parameter dictionary passed into the params argument
           .cv: int or cross validation split generator object
           .scoring: string designating the scoring strategy
           .verbose: flag determining whether progress is printed during fit
           .best_cv_: the list of cross validation scores for the best combination of parameters
           .best_score_: the mean of the best cross validation scores during the fit
           .best_params_: the best parameters (by mean cross validation score)
           .best_estimator_: the estimator set to the best parameters.
        Methods:
            .fit(X,y)
                fits Course_GridSearchCV object on data using cross_validation to save the best set of parameters accoring to the mean of the cross validation scores.
        
        """
        from sklearn.model_selection import ParameterGrid

        self.estimator = estimator
        self.paramgrid = ParameterGrid(params)
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
    
    def fit(self,X,y):
        """
        .fit(X,y)
        ---
        fits Course_GridSearchCV object on data using cross_validation to save the best set of parameters accoring to the mean of the cross validation scores.
        ---
        X: pandas.DataFrame of features
        y: array-like of labels
        """
        self.best_score_ = 0
        for params in self.paramgrid:
            if self.verbose:
                print('trying:')
                print(params)
            self.estimator.set_params(**params)
            scores = course_cross_validate(self.estimator, X, y, scoring=self.scoring, cv=self.cv)
            new_score = np.mean(scores)
            if self.verbose:
                print('average score: ', new_score)
            if new_score > self.best_score_:
                self.best_cv_ = scores
                self.best_score_ = new_score
                self.best_params_ = params
                self.best_estimator_ = self.estimator

    
    
def plot_confusion(y_true, y_pred, encoder=None, labels = None, ax=None, cmap='Greens', save_path = None):
    """
    plot_confusion(y_true, y_pred, encoder=None, labels = None, ax=None, cmap='RdYlBu', save_path = None)
    ---
    plots a confusion matrix using the provided y_true and y_pred using sklearn.metrics.confusion_matrix and
    seabborn.heatmap.
    ---
    y_true: true labels
    y_pred: predicted labels
    encoder: (optional) encoder that encoded the labels.  If provided it will be used to decode the labels to 
    use as xticks and yticks in the confusion matrix.
    labels: (optional) the decoded labels in the order to be displayed on the matrix
    ax: (optional) a matplotlib.pyplot axis object to plot the confusion matrix onto, if desired.
    cmap: (optional) the colormap of the desired confusion matrix
    save_path: (optional) the filepath to save the figure at, if desired.
    """ 
    from seaborn import heatmap
    from sklearn.metrics import confusion_matrix
    from matplotlib.pyplot import savefig
    if encoder:
            y_true = encoder.inverse_transform(y_true)
            y_pred = encoder.inverse_transform(y_pred)
    elif encoder:
        if not labels:
            labels = encoder.classes_
    if not labels:
        labels = np.unique(y_true)
    matrix = heatmap(confusion_matrix(y_true, y_pred, normalize = 'true', labels = labels), 
                    annot = True, cmap = cmap, xticklabels = labels, yticklabels = labels, ax=ax)
    if save_path:
        savefig(save_path)
    return matrix
    
def score_grid(grid,X_val,y_val, labels = None, save_path=None, cmap='Greens'):
    from sklearn.metrics import accuracy_score
    """
    score_grid(grid,X_val,y_val, labels = None, save_path=None, cmap='Greens')
    ---
    Takes a fitted GridSearchCV object, prints the grid.best_score_, uses grid.best_estimator_ to 
    plot a confusion matrix for the validation set provided using plot_confusion(), and returns 
    grid.best_estimator_.
    ---
    grid: fitted sklearn.model_selection.GridSearchCV() object.
    X_val: validation features
    y_val: validation targets
    labels: (optional) order of labels to present in the confusion matrix
    save_path: (optional) path to save confusion matrix image
    cmap: (optional) color map of returned confusion matrix.
    """
    import seaborn as sns
    from src.functions import plot_confusion
    score = grid.best_score_
    model = grid.best_estimator_
    print('best model')
    print(model)
    print('best cv')
    print(grid.best_cv_)
    print('cross validated accuracy score:')
    print(score)
    y_pred = model.predict(X_val)
    print('validation accuracy: ')
    print(accuracy_score(y_val, y_pred))
    print('validation set confusion matrix')
    plot_confusion(y_val, y_pred, labels=labels, save_path=save_path, cmap=cmap)
    return model

def dist_by_course(regs, column):
    import matplotlib.pyplot as plt
    import seaborn as sns
    """
    dist_by_course(regs, column)
    ---
    Takes the name of a column as a string, then creates a histogram of the distribution 
    of values for that column for each course and a KDE plot comparing them all.
    ---
    regs: pandas.DataFrame of data
    column: string, column to plot
    """
    if regs[column].dtype == 'object':
        return "Please choose a numerical column"
    fig, axes = plt.subplots(4,2, figsize=(10,20), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, course in enumerate(regs['code_module'].unique()):
        course_plot = regs[regs['code_module'] == course]
        sns.histplot(data=course_plot, x=column, 
                    palette='husl', stat='density',
                    ax = axes[i]).set_title(f'Distribution of {column} in Course {course}')
    plt.show()
    sns.kdeplot(data=regs, x=column, hue='code_module', common_norm = False,
                 palette='husl').set_title(f'Comparative Distribution of {column} Between Courses')
    
def registration_correlations(save_path = None, columns = None, prediction_window=None, 
                              scaled=False, cmap='coolwarm'):
    """
    registration_correlations(save_path = None, columns = None, prediction_window=None, 
                              scaled=False, drop_course=False, cmap='coolwarm')
    ---
    Loads registrations according to giving prediction window, and creates a 
    dython.associations() correlation plot between all or listed columns.
    ---
    save_path: (optional) path to save figure to.
    columns: (optional) columns to plot correlations between
    prediction_window: (int) how far into the course the dataframe should include
    scaled: (boolean) whether to use CourseScaler to scale data by course.
    cmap: (default is 'coolwarm') colormap for plotted correlations
    """
    
    df = load_OU_data(prediction_window=prediction_window)
    df.loc[df['final_result'] == 'Withdrawn', 'final_result'] = 0
    df.loc[df['final_result'] == 'Fail', 'final_result'] = 1
    df.loc[df['final_result'] == 'Pass', 'final_result'] = 2
    df.loc[df['final_result'] == 'Distinction', 'final_result'] = 3
    
    df.loc[df['age_band'] == '0-35', 'age_band'] = 0
    df.loc[df['age_band'] == '35-55', 'age_band'] = 1
    df.loc[df['age_band'] == '55<=', 'age_band'] = 2
    
    to_drop = ['code_presentation','id_student','module_presentation_length']
    if prediction_window == None:
        to_drop.append('date_unregistration')
    variables = df.drop(columns = to_drop)
    if scaled:
        cs = CourseScaler(drop_course=drop_course)
        variables = cs.fit_transform(variables)
    variables['code_module'] = variables['code_module'].astype('str')
    if type(columns) == list:
        variables = variables[columns]
    fig, ax = plt.subplots(1,1, figsize = (len(variables.columns)*2**1.2, 
                                           len(variables.columns)*1.5**1.2))
    fig.suptitle('Variable Correlations', fontsize = len(variables.columns)*2+5)
       
    associations(variables, ax=ax, mark_columns=False, cmap = cmap)
    if type(save_path) == str:
        fig.savefig(save_path, dpi=250)
    plt.show()