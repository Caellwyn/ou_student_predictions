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
    
    Then it: 
    * drops rows with missing and suspicious data, 
    * derives features: number of days studied, number of activities engaged, and number of clicks for each registration over the course of the module.  Also derives assessment score average and number of assessments completed within the prediction window.
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
    if prediction_window:
        withdrawn_after_predict = (full_registrations['final_result'] == 'Withdrawn') \
                                    & (full_registrations['date_unregistration'] > prediction_window)
    else:
        withdrawn_after_predict = (full_registrations['final_result'] == 'Withdrawn') \
                                    & (full_registrations['date_unregistration'] > 0)
    full_registrations = full_registrations[not_withdrawn | withdrawn_after_predict]
    full_registrations['date_unregistration'].fillna( \
        full_registrations['module_presentation_length'], inplace = True)
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
    Standardizes 'days_studied','activities_completed','total_clicks','assessments_completed', and 'average_score' if they are present in the dataframe.
    methods:
    .fit(X) fits CourseScaler instance to dataframe
    .transform(X) transforms X if fit.
    .fit_transform(X)
    """
    def __init__(self, drop_course=True):
        self.drop_course = drop_course
        pass
    
    def fit(self,X,y=None):
        import pandas as pd
        self.cols = ['days_studied','activities_engaged','total_clicks',\
                     'assessments_completed','average_assessment_score']
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
            if self.drop_course:
                scaled_X.drop('code_module', axis = 1, inplace = True)
            return scaled_X

    
def fail_recall(y_true,y_pred):
    from sklearn.metrics import recall_score
    """
    uses sklearn.metrics.recall_score() to return the recall score of
    non functioning wells.
    Takes single column array or dataframe of true labels
    and one of predictions 
    returns recall score of class 0.
    """
    
    return recall_score(y_true,y_pred,average=None)[0]