import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
from IPython.display import display, clear_output
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
average_precision_score, roc_auc_score, plot_precision_recall_curve, plot_roc_curve, plot_confusion_matrix 



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

    full_registrations = pd.merge(full_registrations, days_studied, on=index_columns, how='left')
    full_registrations = pd.merge(full_registrations, total_activities, on=index_columns, how='left')
    full_registrations = pd.merge(full_registrations, clicks, on=index_columns, how='left')


    # Assessments
    assessments = pd.merge(student_assessments, assessments_info, how='left', on='id_assessment')
    assessments.dropna(subset = ['score'], inplace=True)
    assessments.drop(columns = ['is_banked', 'weight'], inplace=True)
    if prediction_window:
        assessments = assessments[assessments['date_submitted'] <= prediction_window]
    num_assessments = assessments.groupby(by = index_columns).count().reset_index()
    total_assessments = assessments_info.groupby(by=['code_module','code_presentation']).count().reset_index()
    total_assessments = total_assessments.drop(columns = ['assessment_type','date','weight'])
    total_assessments = total_assessments.rename(columns = {'id_assessment':'total_assessments'})
    num_assessments = pd.merge(num_assessments, total_assessments, 
                           how = 'left', on = ['code_module','code_presentation'])
    num_assessments['id_assessment'] = num_assessments['id_assessment']/num_assessments['total_assessments']
    num_assessments.drop(columns = ['total_assessments','date_submitted','score','date','assessment_type'], inplace=True)
    num_assessments
    avg_score = assessments.groupby(by = index_columns).mean().reset_index()
    avg_score.drop(columns = ['date_submitted','id_assessment','date'], inplace=True)

    full_registrations = pd.merge(full_registrations, num_assessments, on=index_columns, how='left')
    full_registrations = pd.merge(full_registrations, avg_score, on=index_columns, how='left')
    full_registrations = full_registrations.fillna(value=0)

    # Rename columns
    new_cols = {'id_assessment':'assessments_completed',
            'score':'average_assessment_score','date':'days_studied',
            'id_site':'activities_engaged','sum_click':'total_clicks'}
    full_registrations = full_registrations.rename(columns = new_cols)
    full_registrations = full_registrations[(full_registrations['total_clicks'] < 4035) 
                          & (full_registrations['activities_engaged'] < 1135)]
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
        self.cols = X.select_dtypes(include = 'number').columns
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
                    self.means.loc[module, col] = course_X[col].mean()
                    self.stds.loc[module, col] = course_X[col].std() + 1e-10
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
            columns = course_X.columns
            course_X, course_y = SMOTE(random_state=111).fit_resample(course_X,course_y)
            course_X = pd.DataFrame(course_X, columns = columns)
            if not drop_course:
                course_X['code_module'] = module            
            course_y = pd.Series(course_y)
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
    y = np.array(y)

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

def cross_val_presentation(model, df, scoring = 'accuracy', verbose = 0):
    import pandas as pd
    index = df.groupby(by=['code_module','code_presentation']).count().index
    scores = pd.DataFrame(columns = ['score'], index = index)
    for module in df['code_module'].unique():
        df_course = df[df['code_module'] == module]
        for presentation in df_course['code_presentation'].unique():
            train = df[(df['code_module'] != module) | (df['code_presentation'] != presentation)]
            test = df[(df['code_module'] == module) & (df['code_presentation'] == presentation)]
            if module in train['code_module'].unique():
                score = model_evaluate_presentation(model, train, test, scoring='accuracy')
                scores.loc[(module,presentation), 'score'] = score
                if verbose > 1:
                    print((module,presentation))
            else:
                scores.loc[(module,presentation), 'score'] = np.nan
    if verbose > 0:
        display(scores.score.mean())
    return scores
    
def model_evaluate_presentation(model, train, test, scoring='accuracy'):
    from sklearn.metrics import accuracy_score, f1_score
    
    X_train = train.drop(columns = ['final_result', 'code_presentation'])
    y_train = train['final_result']
    X_test = test.drop(columns = ['final_result', 'code_presentation'])
    y_test = test['final_result']

    transformed_big_data = process_courses(X_train, y_train, X_test, y_test)
    X_train_transformed, y_train_transformed, X_test_transformed, y_test = transformed_big_data
    model.fit(X_train_transformed, y_train_transformed)
    y_pred = model.predict(X_test_transformed)
    if scoring == 'accuracy':
        return accuracy_score(y_test, y_pred)
    if scoring == 'f1':
        return f1_score(y_test, y_pred, pos_label = 'Needs Intervention')
    
class GridSearchPresentationCV():
    """
    GridSearchPresentationCV(self, estimator, params, scoring='accuracy', verbose = 0)
    ---
    Grid search object using cross validation to find the best combination of parameters and keeping the best estimator and it's best cv scores and the mean of that list of cv scores.  It uses course_cross_validate to to cross validate each model using a custom and static pipeline to appropriately scale and smote each fold by course.
    ---
    estimator: an estimator with .fit(X,y) and .predict(X) methods.
    params: a dictionary of estimator hyperparameters of the format {'parameter_name':string or list of parameter values}
    cv: (optional) an int or cross validation generator object (like KFold), default = 5
    scoring: (optional) either 'accuracy' or 'f1_score'. Default = 'accuracy'
    verbose: (optional) if True, prints each attempted combination of parameter and their cross validation scores.
    """
    def __init__(self, estimator, params, scoring='accuracy', verbose = 0):
        """
        GridSearchPresentationCV(self, estimator, params, cv=5, scoring='accuracy', verbose = False)
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
        self.scoring = scoring
        self.verbose = verbose
    
    def fit(self, df, show_progress = False):
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
            scores = cross_val_presentation(self.estimator, df, scoring=self.scoring)
            new_score = np.mean(scores.score)
            if self.verbose == 1:
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
        savefig(save_path, dpi=250)
    return matrix
    
def score_grid(grid,X_val,y_val, labels = None, save_path=None, cmap='Greens', scoring='accuracy'):
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
    from sklearn.metrics import accuracy_score, f1_score
    from IPython.display import display
    score = grid.best_score_
    model = grid.best_estimator_
    print('best model')
    print(model)
    print('best cv')
    display(grid.best_cv_)
    print('cross validated accuracy score:')
    print(score)
    y_pred = model.predict(X_val)
    print(f'validation {scoring}: ')
    if scoring == 'accuracy':
        print(accuracy_score(y_val, y_pred))
    if scoring == 'f1':
        print(f1_score(y_val, y_pred))
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
    
def registration_correlations(passed_df=None, save_path = None, columns = None, prediction_window=None, 
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
    import matplotlib.pyplot as plt
    from dython.nominal import associations
    if type(passed_df) == type(None):
        df = load_OU_data(prediction_window=prediction_window)
    else: df = passed_df.copy()
    if 'final_result' in df.columns:
        df.loc[df['final_result'] == 'Withdrawn', 'final_result'] = 0
        df.loc[df['final_result'] == 'Fail', 'final_result'] = 1
        df.loc[df['final_result'] == 'Pass', 'final_result'] = 2
        df.loc[df['final_result'] == 'Distinction', 'final_result'] = 3
    if 'age_band' in df.columns:
        df.loc[df['age_band'] == '0-35', 'age_band'] = 0
        df.loc[df['age_band'] == '35-55', 'age_band'] = 1
        df.loc[df['age_band'] == '55<=', 'age_band'] = 2
    
    to_drop = ['code_presentation','id_student','module_presentation_length']
    if prediction_window == None:
        to_drop.append('date_unregistration')
    for column in to_drop:
        if column in df.columns:
            df = df.drop(column, axis = 1)
    if scaled:
        if 'code_module' in df.columns:
            cs = CourseScaler(drop_course = False)
            df = cs.fit_transform(df)
        else:
            print('cannot scale, code_module not found in columns')
    if type(columns) == list:
        df = df[columns]
    fig, ax = plt.subplots(1,1, figsize = (len(df.columns)*2**1.2, 
                                           len(df.columns)*1.5**1.2))
    fig.suptitle('Variable Correlations', fontsize = len(df.columns)*2+5)
       
    associations(df, ax=ax, mark_columns=False, cmap = cmap)
    if type(save_path) == str:
        fig.savefig(save_path, dpi=250)
    plt.show()
    
def graph_model_history(history, metric = 'acc'):
    """
    Takes as argument a model history: either the return of a KerasClassifier or the model.history of a Keras model.
    Plots model accuracy, validation accuracy, model loss, and validation loss training histories, if they are present in the history.history dictionary object:
    """
        
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1,2, figsize = (10,5))
    if 'loss' in history.history.keys():
        axes[0].plot(history.history['loss'], label = 'Train Loss')
    if 'val_loss' in history.history.keys():
        axes[0].plot(history.history['val_loss'], label = 'Validation Loss')
    axes[0].legend()
    if metric in history.history.keys():
        axes[1].plot(history.history[metric], label = f'Train {metric}')
    if f'val_{metric}' in history.history.keys():
        axes[1].plot(history.history[f'val_{metric}'], label = f'Validation {metric}')
    axes[1].legend()

    plt.show()
    
# Timeseries Tools

def test_model(model, X_val, y_val, plot_confusion=False):
    """Takes a fitted model and validation data and return None.
    Displays a confusion matrix and a model score (default scoring method for model)"""
    from sklearn.metrics import plot_confusion_matrix
    import matplotlib.pyplot as plt
    print('model: ', model)
    print(model.score(X_val, y_val))
    if plot_confusion:
        plot = plot_confusion_matrix(model, X_val, y_val, 
                             normalize='true',
                             cmap = 'Greens')
    plt.show()
    
def add_model(model, X_t, y_t, X_val, y_val,
              preprocessing=None, 
              features='Not Provided'):
    """
    Scores a model by several metrics and saves it to a hyperparameter table
    """
    train_probs = model.predict_proba(X_t)[:,1]
    train_yhat = np.round_(train_probs)
    val_probs = model.predict_proba(X_val)[:,1]
    val_yhat = np.round_(val_probs)
    parameters = {'model' : type(model).__name__,
                  'val_roc_auc' : roc_auc_score(y_val, val_probs), 
                  'train_roc_auc': roc_auc_score(y_t, train_probs), 
                  'val_accuracy': accuracy_score(y_val, val_yhat),
                  'train_accuracy': accuracy_score(y_t, train_yhat),
                  'val_f1_score': f1_score(y_val, val_yhat),
                  'train_f1_score': f1_score(y_val, val_yhat),
                  'features': features,
                  'preprocessing': preprocessing,
                 }
    parameters.update(model.get_params())
    parameters = pd.DataFrame(parameters, index=[0])

    try:
        table = pd.read_csv('hyperparameter_table.csv')
    except:
        table = pd.DataFrame()
    table = table.append(parameters, ignore_index=True)
    table = table.drop_duplicates(subset=table.columns[7:], keep='last')
    table = table.sort_values(by='val_roc_auc', ascending=False)
    table.to_csv('hyperparameter_table.csv', index=False)
    

def add_hypersearch(opt):
    
    """
    opt: a fitted hyperparameter search instance.
    Adds scores and hyperparameters from a hyperparameter search
    to a hyperparameter table.
    """
    model = opt.best_estimator_
    scores = {'model':type(model).__name__,
               'val_roc_auc': opt.cv_results_['mean_test_score'],
               'train_roc_auc': opt.cv_results_['mean_train_score'],
              }
    scores = pd.DataFrame(scores)
    
    parameters = pd.DataFrame(opt.cv_results_['params'])
    table = pd.concat([scores, parameters], axis=1)

    try:
        old_table = pd.read_csv('hyperparameter_table.csv')
    except:
        old_table = pd.DataFrame()
    table = old_table.append(table, ignore_index=True)
    table = table.drop_duplicates()
    table = table.sort_values(by='val_roc_auc', ascending=False)
    table.to_csv('hyperparameter_table.csv', index=False)

def get_timeseries_table(prediction_window=None, binary_labels=False, one_hot_modules=False):


    """
    Takes prediction_window (int), which is the day of the course you want to stop taking data make your prediction.
    Returns a table, size = (number of registrations, prediction_window * 3 + number of assessments * 2 + 4)
    table includes count of activities, clicks, and clicks*activities for each day of the course in the window,
    relative date submitted and score for each assessment taken, student registration, module code, 
    and final course outcome (target).

    Set binary_lables to True to change 'Pass' and 'Distinction' to 0 and 'Fail' and 'Withdrawn' to 1
    
    Set one_hot to true to one-hot encode the module codes.
    """
    student_vle, assessments, assessment_info, student_info, student_unregistration = import_tables(prediction_window)
    
    merged_activities = merge_activity_tables(prediction_window, student_vle,  student_info, student_unregistration)
    
    activity_df = get_activity_df(prediction_window, merged_activities)
    
    assessment_df = get_assessment_df(prediction_window, assessments, assessment_info)
    
    if len(assessment_df) > 0:
        datatable = pd.merge(assessment_df, activity_df, how='outer', on='registration')

    else:
        print('No assessments found in this prediction window, table only includes activities and clicks')
        datatable = activity_df

    datatable = datatable.fillna(0)
    if prediction_window:
        datatable = datatable[datatable['date_unregistration'] >= prediction_window]
    datatable = datatable.drop(columns=['date_unregistration'])
    if binary_labels:
        binary_labels = {'Pass':0,
                         'Distinction':0,
                         'Withdrawn':1,
                         'Fail':1}

        datatable['final_result'] = datatable['final_result'].map(binary_labels)
    if one_hot_modules:
        datatable = pd.get_dummies(datatable, prefix='module', columns=['code_module'])
    datatable = datatable.set_index('registration')
    return datatable

def import_tables(prediction_window):
    """
    Loads necessary tables from anonymisedData.zip
    """
    import pandas as pd
    import numpy as np
    import zipfile

    #load data
    zf = zipfile.ZipFile('../content/anonymisedData.zip') 
    student_vle = pd.read_csv(zf.open('studentVle.csv'))
    if prediction_window:
        student_vle = student_vle[student_vle['date'] < prediction_window]
    else: 
        student_vle = student_vle[student_vle['date'] < 270]
    
    assessments = pd.read_csv(zf.open('studentAssessment.csv'), skiprows=[128223,64073])
    assessments_info = pd.read_csv(zf.open('assessments.csv'))
    
    student_info =  pd.read_csv(zf.open('studentInfo.csv'),
                               usecols = ['code_module','code_presentation','id_student',
                                         'final_result'])
    student_unregistration = pd.read_csv(zf.open('studentRegistration.csv'),
                                  usecols = ['code_module','code_presentation','id_student',
                                             'date_unregistration'])

    #combine module, presentation, and student id columns into one registration column
    student_vle['registration'] = student_vle['code_module'] \
                                    + student_vle['code_presentation'] \
                                    + student_vle['id_student'].astype(str)

    student_info['registration'] = student_info['code_module'] \
                                + student_info['code_presentation'] \
                                + student_info['id_student'].astype('str')

    student_unregistration['registration'] = student_unregistration['code_module'] \
                                + student_unregistration['code_presentation'] \
                                + student_unregistration['id_student'].astype('str')
    
    student_unregistration['date_unregistration'].fillna(student_vle['date'].max()+1, inplace=True)
    
    return (student_vle, assessments, assessments_info, student_info, student_unregistration)

def merge_activity_tables(prediction_window, student_vle, student_info, student_unregistration):
    """
    helper function for get_table(). Takes several data tables and returns dataframe of the 
    numbers of activities and clicks and dates of withdrawal and course results.  
    Each row represent one day of work for each student registration up the the prediction window.
    """
    #group by registration and day
    vle_group = student_vle.groupby(by = ['registration', 'date'])

    #sum activities and clicks per day. activities are '.count()' because each row is an activity.
    sum_activities = vle_group.count().reset_index()[['registration','date','id_site']]
    sum_clicks = vle_group.sum().reset_index()[['registration','date','sum_click']]

    merged_activities = pd.merge(sum_activities, sum_clicks, on=['registration','date'], how='inner',
                                 validate='1:1')
    merged_activities = merged_activities.merge(student_info[['registration','final_result', 'code_module']], 
                                                on='registration')
    merged_activities = merged_activities.rename(columns = {'id_site':'sum_activities'})

    merged_activities['activities_x_clicks'] = merged_activities['sum_activities'] \
                                             * merged_activities['sum_click']
    merged_activities = merged_activities.sort_values(by=['registration','date'])

    #A little more cleanup
    merged_activities = merged_activities.merge(student_unregistration[['registration','date_unregistration']],
                                                                      on='registration', how='left')
    
    if prediction_window:
        merged_actitivies = merged_activities['date_unregistration'].fillna(prediction_window)
    else:
        merged_actitivies = merged_activities['date_unregistration'].fillna(merged_activities['date'].max())
    merged_activities = merged_activities.fillna(0)
    merged_activities = merged_activities.drop_duplicates(keep='first')
    return merged_activities

def get_activity_df(prediction_window, merged_activities):
    """
    Takes the merged table of activities and returns a new table with one row per student registration
    and columns for numbers of activities, clicks, and activities * clicks for each day of the course
    up to the prediction window
    """

    if prediction_window:
        date_range = range(merged_activities.date.min(), prediction_window)
    else:
        date_range = range(merged_activities.date.min(), merged_activities.date.max())

    activity_df = pd.DataFrame()
    activity_df['registration'] = merged_activities['registration'].unique()
    counter = len(date_range)

    for date in date_range:
        single_date_df = merged_activities[merged_activities['date'] == date][['registration',
                                                                               'sum_activities',
                                                                               'sum_click',
                                                                               'activities_x_clicks']]

        single_date_df.columns = ['registration'] + [f'{x}_{date}' for x in single_date_df.columns[1:]]

        activity_df = activity_df.merge(single_date_df, 
                                                how='left', 
                                                on='registration',
                                                validate = '1:m')

        print('activity days merged: ', counter)
        clear_output(wait=True)
        counter -= 1

    activity_df = activity_df.fillna(0)

    activity_df = activity_df.merge(merged_activities[['registration','code_module','final_result',
                                                       'date_unregistration']].drop_duplicates(), 
                                    how='left', 
                                    on='registration')

    if prediction_window:
        activity_df = activity_df[activity_df['date_unregistration'] >= prediction_window]

    return activity_df

def get_assessment_df(prediction_window, assessments, assessment_info):
    """
    Merges student assessments with assessment information.
    Returns a dataframe with a row for each student registration
    and columns for each assessment students completed before the prediction window 
    """
    full_assess = assessments.merge(assessment_info, on='id_assessment', how='left')
    full_assess = full_assess.dropna(axis=0, subset=['score'])
    full_assess['date'] = full_assess['date'].fillna(full_assess['date_submitted'])

    full_assess['registration'] = full_assess['code_module'] \
                                + full_assess['code_presentation'] \
                                + full_assess['id_student'].astype(int).astype(str)
    full_assess['assess_submitted'] = full_assess['date_submitted'] - full_assess['date']
    full_assess = full_assess[['registration','assess_submitted','score','date']]

    if prediction_window:
        full_assess = full_assess[full_assess['date'] < prediction_window]

    if len(full_assess) > 0:
        grouped_assess = full_assess.groupby('registration')
        max_assess_count = grouped_assess.count().max().max()
    else:
        return full_assess

    counter = max_assess_count
    temp_assess = full_assess.sort_values(by=['date','assess_submitted'])
    registered = full_assess.registration.unique()
    assess_timeseries = pd.DataFrame(registered, columns=['registration'])

    for assess_num in range(max_assess_count):
        single_assess = temp_assess.groupby('registration').head(1)
        single_assess = single_assess[['registration','assess_submitted','score']]
        single_assess = single_assess.rename(columns = {'assess_submitted':f'assess_submitted_{assess_num+1}',
                                                              'score':f'assess_score_{assess_num+1}'})
        assess_timeseries = assess_timeseries.merge(single_assess, on='registration', how='left')
        temp_assess = temp_assess.drop(index = single_assess.index)

        print('assessments merged: ', counter)
        clear_output(wait=True)
        counter -= 1

    return assess_timeseries