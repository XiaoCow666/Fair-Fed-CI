import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_core import FairAttentionMLP

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class FairEduSystem:
    # ... (Previous __init__ and _load_system)
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.df = None
        self.feature_cols = []
        self._load_system()

    def _load_system(self):
        # ... (Same as before)
        print("Initializing Fair-Edu System...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data not found at {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        exclude_cols = ['JMXH', 'avg_score', 'sensitive_attribute']
        self.feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        print(f"Loaded {len(self.df)} students with {len(self.feature_cols)} features.")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        self.model = FairAttentionMLP(input_dim=len(self.feature_cols))
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        print("Model loaded successfully.")

    def predict_student(self, student_id):
        # ... (Same logic, but return full feature importance for plotting)
        student_row = self.df[self.df['JMXH'] == student_id]
        if student_row.empty:
            return None, "Student not found."
            
        features = student_row[self.feature_cols].values.astype(float)
        x = torch.tensor(features, dtype=torch.float32)
        
        with torch.no_grad():
            score, attn_weights = self.model(x)
            
        score_val = score.item()
        attn_weights = attn_weights.numpy().flatten()
        
        threshold = 0.6
        is_pass = score_val >= threshold
        
        feature_importance = []
        for i, col in enumerate(self.feature_cols):
            # Only keep Course Features (CJ_) for clarity in report
            if col.startswith("CJ_"):
                feature_importance.append({
                    'feature': col.replace("CJ_", ""),
                    'weight': attn_weights[i],
                    'value': features[0][i]
                })
        
        # Sort by weight
        feature_importance.sort(key=lambda x: x['weight'], reverse=True)
        
        result = {
            "student_id": student_id,
            "predicted_score": score_val * 100,
            "prediction": "及格 (Pass)" if is_pass else "不及格/高风险 (Fail/Risk)",
            "risk_level": "Low" if score_val > 0.7 else ("Medium" if score_val > 0.6 else "High"),
            "explanation": feature_importance[:3], # Top 3 for text
            "full_explanation": feature_importance # All for plot
        }
        return result, None

    def generate_plot(self, result, output_path="student_report.png"):
        """Generate a visual report for the student."""
        data = result['full_explanation'][:10] # Top 10 features
        
        df_plot = pd.DataFrame(data)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='weight', y='feature', data=df_plot, palette='viridis')
        plt.title(f"学生 {result['student_id']} 学业预警归因分析 (Top 10 Factors)")
        plt.xlabel("影响力权重 (Attention Weight)")
        plt.ylabel("课程名称")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
        plt.close()

import argparse
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "client_0_model.pth")
    
    parser = argparse.ArgumentParser(description="Fair-Edu Prediction System")
    parser.add_argument("--data_path", type=str, default=os.path.join(base_dir, "data", "processed_data_v2.csv"), help="Path to student data CSV")
    parser.add_argument("--student_id", type=str, help="Specific Student ID to predict")
    parser.add_argument("--plot", action="store_true", help="Generate visualization plot")
    args = parser.parse_args()
    
    try:
        system = FairEduSystem(model_path, args.data_path)
    except Exception as e:
        print(f"System Init Failed: {e}")
        return

    def print_result(res):
        print(f"\n[Prediction Result]")
        print(f"Student ID: {res['student_id']}")
        print(f"Prediction: {res['prediction']}")
        print(f"Score: {res['predicted_score']:.1f} / 100")
        print(f"Risk Level: {res['risk_level']}")
        print("Key Factors (Top 3 Influential Courses):")
        for item in res['explanation']:
            print(f"  - {item['feature']}: Importance={item['weight']:.4f}, Student Score={item['value']*100:.1f}")

    # Mode 1: Single Prediction via CLI
    if args.student_id:
        res, err = system.predict_student(args.student_id)
        if err:
            print(f"Error: {err}")
        else:
            print_result(res)
            if args.plot:
                system.generate_plot(res, f"report_{args.student_id}.png")
        return

    # Mode 2: Interactive Loop
    print("\n--- Fair-Edu Interactive System ---")
    print("Enter Student ID to predict (or 'q' to quit, 'list' to see 5 random IDs)")
    
    while True:
        user_input = input("\nStudent ID > ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
            
        if user_input.lower() == 'list':
            print("Random available IDs:", system.df['JMXH'].sample(5).tolist())
            continue
            
        if not user_input:
            continue
            
        res, err = system.predict_student(user_input)
        if err:
            print(f"Error: {err}")
        else:
            print_result(res)
            # Always plot in interactive mode for better UX? Or ask?
            # Let's just save it as 'current_student_report.png' to avoid clutter
            system.generate_plot(res, "current_student_report.png")

if __name__ == "__main__":
    main()
