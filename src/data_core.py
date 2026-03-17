"""
Fair-Fed-CI: Privacy-Preserving Educational Prediction via Federated Learning
-------------------------------------------------------------------------
Data Core Module: Responsible for loading, cleaning, and normalizing 
multi-college student records into non-IID federated clients.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

class DataCore:
    """
    Fair-Fed-CI v2 Data Core
    Responsible for loading, cleaning, and strictly normalizing student data.
    """
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.df = None
        self.feature_cols = []
        self.target_col = 'target_score'
        self.sensitive_col = 'sensitive_attribute'
        
        # Normalizers
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_and_clean(self):
        """Load raw data and perform initial cleaning with Temporal Split."""
        print(f"Loading data from {self.raw_data_path}...")
        # Use index_col=0 to handle the unnamed index column
        self.df = pd.read_csv(self.raw_data_path, encoding='gbk', index_col=0)
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip().str.replace('"', '')
        print(f"Columns: {self.df.columns.tolist()}")
        
        # 1. Clean Scores
        self.df[['clean_score', 'is_retake', 'is_absent']] = self.df.apply(
            self._clean_score_row, axis=1, result_type='expand'
        )
        
        # 2. Extract Temporal Info (Semester Index)
        # Use apply for robust parsing
        temporal_info = self.df.apply(self._parse_temporal_info, axis=1, result_type='expand')
        self.df[['course_year', 'xq']] = temporal_info
        
        # Drop rows with invalid year info
        before_drop = len(self.df)
        self.df.dropna(subset=['course_year', 'xq'], inplace=True)
        print(f"Dropped {before_drop - len(self.df)} rows with invalid temporal info.")
        
        # Calculate enroll_year as the minimum course_year per student
        enroll_years = self.df.groupby('JMXH')['course_year'].min().reset_index()
        enroll_years.rename(columns={'course_year': 'enroll_year'}, inplace=True)
        self.df = pd.merge(self.df, enroll_years, on='JMXH', how='left')
        
        # Calculate semester index
        self.df['semester_idx'] = (self.df['course_year'] - self.df['enroll_year']) * 2 + (self.df['xq'] - 1)
        
        # Filter valid semesters (0 to 7 for 4 years)
        self.df = self.df[(self.df['semester_idx'] >= 0) & (self.df['semester_idx'] <= 7)]
        
        print("Temporal split configuration: Input <= Sem 3 (Year 1-2), Target >= Sem 4 (Year 3+)")
        
        # 3. Split Data
        # Input Data: Semesters 0, 1, 2, 3
        input_df = self.df[self.df['semester_idx'] <= 3].copy()
        
        # Target Data: Semesters 4, 5, 6, 7
        target_df = self.df[self.df['semester_idx'] >= 4].copy()
        
        # Add a flag for failed courses (不及格课程数)
        input_df['is_failed'] = (input_df['clean_score'] < 60).astype(int)
        
        # 4. Aggregate Input Features (Student History)
        agg_funcs = {
            'clean_score': ['mean', 'std', 'max', 'min'],
            'KCMC': 'count',
            'is_retake': 'sum',
            'is_absent': 'sum',
            'is_failed': 'sum'
        }
        student_features = input_df.groupby('JMXH').agg(agg_funcs)
        student_features.columns = [f"{col[0]}_{col[1]}" for col in student_features.columns]
        student_features = student_features.reset_index()
        
        student_features.rename(columns={
            'clean_score_mean': 'avg_score', # This is now avg score of Year 1-2
            'clean_score_std': 'std_score',
            'clean_score_max': 'max_score',
            'clean_score_min': 'min_score',
            'KCMC_count': 'total_courses',
            'is_retake_sum': 'retake_count',
            'is_absent_sum': 'absent_count',
            'is_failed_sum': 'failed_count'
        }, inplace=True)
        student_features.fillna(0, inplace=True)
        
        # 5. Calculate Target Label (Future Performance)
        # Target = Average score of Year 3+
        target_labels = target_df.groupby('JMXH')['clean_score'].mean().reset_index()
        target_labels.rename(columns={'clean_score': 'target_score'}, inplace=True)
        
        # Merge Features and Target
        # Inner join: We only keep students who have BOTH history AND future data
        # (i.e., we drop freshmen/sophomores who haven't reached Year 3 yet)
        self.df = pd.merge(student_features, target_labels, on='JMXH', how='inner')
        
        print(f"Students after temporal merge: {len(self.df)} (Only those with Year 1-2 AND Year 3+ data)")
        
        # 6. Pivot Top N Courses (Only from Input Data)
        # Identify top 20 most frequent courses in INPUT data (Year 1-2)
        top_courses = input_df['KCMC'].value_counts().head(20).index.tolist()
        print(f"Top 20 Input Courses (Year 1-2): {top_courses}")
        
        course_df = input_df[input_df['KCMC'].isin(top_courses)][['JMXH', 'KCMC', 'clean_score']]
        course_pivot = course_df.pivot_table(index='JMXH', columns='KCMC', values='clean_score', fill_value=0)
        course_pivot.columns = [f"CJ_{c}" for c in course_pivot.columns]
        
        self.df = pd.merge(self.df, course_pivot, on='JMXH', how='left')
        self.df.fillna(0, inplace=True)
        
        # 7. Merge Static Info
        # Get static info from original raw df (any row for the student is fine)
        static_info = pd.read_csv(self.raw_data_path, encoding='gbk', index_col=0).drop_duplicates(subset='JMXH')[['JMXH', 'XY', 'ZY']]
        self.df = pd.merge(self.df, static_info, on='JMXH', how='left')
        
        return self.df

    def _parse_temporal_info(self, row):
        """Helper to parse JMXH and XNMC."""
        try:
            # XNMC: 2020-2021
            xn_str = str(row['XNMC'])
            if '-' in xn_str:
                course_year = int(xn_str.split('-')[0])
            else:
                # Try to parse as int directly if it's just a year
                if xn_str.isdigit():
                    course_year = int(xn_str)
                else:
                    return np.nan, np.nan
                
            # XQMC: 1 or 2
            xq_str = str(row['XQMC'])
            if xq_str.isdigit():
                xq = int(xq_str)
            else:
                return np.nan, np.nan
            
            return course_year, xq
        except Exception as e:
            # Print first 5 errors
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            if self._error_count < 5:
                print(f"Error parsing row: {row.name}, Error: {e}")
                print(f"Values: JMXH={row.get('JMXH')}, XNMC={row.get('XNMC')}, XQMC={row.get('XQMC')}")
                self._error_count += 1
            return np.nan, np.nan

    def _clean_score_row(self, row):
        """Helper to parse score strings."""
        score_map = {'优秀': 95, '良好': 85, '中等': 75, '及格': 65, '通过': 60, '不及格': 50}
        
        # Try numeric
        score = pd.to_numeric(row['BFZCJ'], errors='coerce')
        if pd.isna(score):
            cj_str = str(row['CJ']).strip()
            score = score_map.get(cj_str, np.nan)
            
        # Flags
        is_retake = 1 if '补考' in str(row['LX']) else 0
        is_absent = 1 if '缺考' in str(row['CJBZ']) or '取消' in str(row['CJBZ']) else 0
        
        # Default NaN score to 0 if absent, else keep NaN to drop later or impute
        if pd.isna(score) and is_absent:
            score = 0.0
            
        return score, is_retake, is_absent

    def preprocess_and_normalize(self):
        """
        Strict normalization pipeline.
        Scales all continuous features to [0, 1].
        One-hot encodes categorical features.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_clean() first.")
            
        print("Starting preprocessing and normalization...")
        
        # 1. Define Feature Groups
        # Add target_score to continuous columns so it gets normalized too
        continuous_cols = ['avg_score', 'std_score', 'max_score', 'min_score', 
                           'total_courses', 'retake_count', 'absent_count', 'failed_count', 'target_score']
        
        # Add Course Columns (CJ_...) to continuous_cols
        course_cols = [c for c in self.df.columns if c.startswith('CJ_')]
        continuous_cols.extend(course_cols)
        
        categorical_cols = ['XY', 'ZY'] # College, Major
        
        # 2. Handle Categorical (One-Hot)
        # We keep 'XY' as sensitive attribute before encoding
        self.df[self.sensitive_col] = self.df['XY']
        
        # One-Hot Encoding
        df_encoded = pd.get_dummies(self.df, columns=categorical_cols, dummy_na=False)
        
        # 3. Normalize Continuous Features
        # CRITICAL: Fit scaler on the whole dataset (or train set in real scenario)
        # Here we scale to [0, 1]
        df_encoded[continuous_cols] = self.scaler.fit_transform(df_encoded[continuous_cols])
        
        # 4. Define Target
        self.df = df_encoded
        self.target_col = 'target_score' 
        
        # Update feature columns (exclude target and sensitive meta)
        self.feature_cols = [c for c in self.df.columns if c not in ['JMXH', self.sensitive_col, self.target_col]]
        
        print("Normalization complete.")
        print(f"Feature columns: {len(self.feature_cols)}")
        print(self.df[continuous_cols].describe()) # Verify range is 0-1
        
        return self.df

    def save_processed(self, output_path):
        """Save the processed dataframe."""
        print(f"Saving data to {output_path}...")
        print(f"Final Columns: {self.df.columns.tolist()}")
        if 'target_score' not in self.df.columns:
            print("CRITICAL ERROR: target_score missing before save!")
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    try:
        # Test run
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_path = os.path.join(base_dir, "data", "raw_data.csv")
        out_path = os.path.join(base_dir, "data", "processed_data_v2.csv")
        
        core = DataCore(raw_path)
        core.load_and_clean()
        core.preprocess_and_normalize()
        core.save_processed(out_path)
    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
