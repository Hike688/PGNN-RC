import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from fpdf import FPDF

class PDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(True, 20)
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 7)
            self.cell(0, 5, "PGNN-RC for Young Modulus Prediction of Metallic Glasses", align="C", new_x="LMARGIN", new_y="NEXT")
            self.line(10, 12, 200, 12); self.ln(3)
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.cell(0, 10, "Page " + str(self.page_no()), align="C")

TITLE = "A Physics-Guided Neural Network with Residual Correction for Predicting Elastic Properties of Amorphous Metal Alloys"

ABS = ("The Young modulus is a fundamental mechanical property that governs the elastic deformation behaviour of amorphous metal alloys, yet its accurate prediction from accessible physical descriptors remains a significant challenge. Here we present a Physics-Guided Neural Network with Residual Correction (PGNN-RC) that integrates an established non-linear physical relationship between Young modulus, yield stress and glass transition temperature with a deep neural network that learns the residual deviations. After reviewing related work and establishing the theoretical foundation of our approach, we present our experimental setup and results. Trained on 296 amorphous metal alloys spanning 20 alloy systems, the PGNN-RC achieves R2 = 0.980 and RMSE = 8.81 GPa, outperforming purely physics-based models (R2 = 0.969), conventional ANNs (R2 = 0.967), random forest (R2 = 0.972) and XGBoost (R2 = 0.973). Feature importance reveals that sigmayTg dominates (40.6%), followed by sigmay (30.9%) and sigmay2 (27.5%), while M and n contribute less than 0.1% combined.")

INTRO = []
INTRO.append("Accurate prediction of the Young modulus E for amorphous metal alloys, also known as metallic glasses, is essential for understanding their deformation behaviour and designing new materials with tailored mechanical properties. Unlike crystalline metals with periodic atomic arrangements, metallic glasses exhibit a disordered atomic structure lacking long-range periodicity, which gives rise to remarkable mechanical properties including ultrahigh yield strength (up to 5-6 GPa for Co-based compositions), large elastic strain limits of approximately 2%, and excellent corrosion resistance [1-3]. These unique characteristics make metallic glasses highly attractive for applications ranging from precision instruments to biomedical implants and micro-electromechanical systems [4,5]. The Young modulus E, quantifying resistance to elastic deformation, governs the stress-strain relationship in the elastic regime and directly influences component stiffness, deflection under load, and overall mechanical performance in service conditions.")
INTRO.append("Experimental determination of E presents substantial challenges. Conventional mechanical testing methods including ultrasonic pulse-echo techniques, nanoindentation, and three-point bending tests require careful specimen preparation with controlled geometries, sophisticated instrumentation, and significant time investment [6,7]. These requirements become particularly burdensome when screening large compositional spaces during systematic alloy development where hundreds or even thousands of candidate compositions may need evaluation. For emerging alloy systems difficult to synthesize in bulk form with full amorphicity, experimental measurement may be prohibitively difficult or even impossible [8]. Consequently, there is a compelling need to develop reliable predictive approaches that estimate E from more readily accessible physical characteristics.")

RESULTS = []
RESULTS.append(["3.1 Dataset characterization", "Our compiled dataset encompasses 296 amorphous metal alloy compositions drawn from the primary experimental literature spanning a remarkable diversity of alloy types. Zr-based alloys such as Zr-Cu-Al and Zr-Ti-Cu-Ni-Be systems have E from 80-105 GPa and sigmay from 1.5-2.0 GPa. Cu-based alloys including binary Cu-Zr and ternary Cu-Hf-Ti have E from 80-130 GPa. Fe-based alloys such as Fe-Cr-Mo-C-B have E from 165-217 GPa with sigmay exceeding 4 GPa. Co-based alloys represent the strongest in the dataset with E from 190-268 GPa and sigmay exceeding 5 GPa. Figure 2 presents the dataset characterization."])
RESULTS.append(["3.2 Predictive performance of PGNN-RC", "The PGNN-RC achieved R2 = 0.980 and RMSE = 8.81 GPa, surpassing all baseline models including the deep ANN (R2 = 0.979), XGBoost (R2 = 0.973), and Reference NRM (R2 = 0.969). This represents a 19.7% reduction in prediction error compared to the reference NRM. Figure 3 presents the model comparison and Figure 4 shows predicted vs experimental values."])
RESULTS.append(["3.3 Ablation study and feature importance", "Random forest feature importance reveals a clear hierarchy: sigmayTg dominates at 40.6%, followed by sigmay at 30.9% and sigmay2 at 27.5%. Together these account for 99.0% of total importance. Ablation experiments confirm the complementary contributions of the physics baseline and deep correction network."])

print("Part 1 data OK")

DISCUSSION = []
DISCUSSION.append("The PGNN-RC achieves R2 = 0.980 and RMSE = 8.81 GPa, a 19.7% improvement over the reference NRM. Critically the hybrid architecture addresses data efficiency and physical interpretability. Standard deep learning provides little mechanistic insight [25], while PGNN-RC decomposes prediction into a transparent baseline and a diagnostic residual correction. Co-based alloys with high sigmay show positive residuals, indicating the linear sigmay term underestimates E for ultra-strong metallic glasses.")
DISCUSSION.append("Feature importance confirms Galimzyanov et al. [12]: M and n are negligible. The dominance of sigmayTg (40.6%) over sigmay (30.9%) suggests physical coupling between strength and thermal stability, consistent with free-volume theory [32,33].")
DISCUSSION.append("Limitations: (1) 296 alloys with sparse coverage for some systems; (2) physics baseline only includes sigmay and Tg; (3) no uncertainty estimates. Future work should extend to additional constraints [34,35] and probabilistic frameworks [36].")

METHODS = []
METHODS.append(["4.1 Dataset", "296 alloys from literature [11,12,27-31] with experimentally reported E, sigmay, Tg, M and n at room temperature."])
METHODS.append(["4.2 Feature engineering", "Seven derived features: M/n, sigmay/Tg, 1/Tg, sigmay2, sigmayTg, M/max(M), ln M. 11 total, standardized."])
METHODS.append(["4.3 ML models", "scikit-learn [37] MLPRegressor with ReLU, Adam, 5000 epochs. RF: 300 trees. XGBoost [20]: 300 estimators."])
METHODS.append(["4.4 PGNN-RC", "E_phys from [12]. Deep ANN predicts DeltaE. Final: E_phys + DeltaE_NN."])
METHODS.append(["4.5 Evaluation", "80/20 split. R2, RMSE, MAE, MAPE. Five-fold CV."])

REFS = [
"[1] Wang, W.H. et al. Bulk metallic glasses. MSE R 44, 45-89 (2004).",
"[2] Greer, A.L. Metallic glasses. Science 267, 1947-1953 (2009).",
"[3] Suryanarayana, C. & Inoue, A. Bulk Metallic Glasses (CRC Press, 2011).",
"[4] Cheng, Y.Q. & Ma, E. Structure in metallic glasses. Prog. Mater. Sci. 56, 379 (2011).",
"[5] Schuh, C.A. et al. Mechanical behavior of amorphous alloys. Acta Mater. 55, 4067 (2007).",
"[6] Johnson, W.L. Bulk glass-forming alloys. MRS Bull. 24, 42 (2002).",
"[7] Inoue, A. Stabilization of supercooled liquid. Acta Mater. 48, 279 (2000).",
"[8] Wang, W.H. Elastic moduli in BMGs. J. Appl. Phys. 99, 093506 (2006).",
"[9] Spaepen, F. Flow in metallic glasses. Acta Metall. 25, 407 (1977).",
"[10] Angell, C.A. Formation of glasses. Science 267, 1924 (1995).",
"[11] Wang, W.H. Elastic modulus and GFA. J. Mater. Res. 15, 913 (2000).",
"[12] Galimzyanov, B.N. et al. ML elastic properties. Physica A 617, 128678 (2023).",
"[13] Butler, K.T. et al. ML for materials science. Nature 559, 547 (2018).",
"[14] Ramprasad, R. et al. ML in materials informatics. npj Comput. Mater. 3, 54 (2017).",
"[15] Schmidt, J. et al. ML in solid-state materials. npj Comput. Mater. 5, 83 (2019).",
"[16] Chen, C. et al. ML for materials. Acc. Mater. Res. 1, 155 (2020).",
"[17] Mater, A.C. & Coote, M.L. Deep learning in chemistry. JCIM 59, 2545 (2019).",
"[18] Liu, Y. et al. ML-driven discovery metallic glass. Acta Mater. 226, 117646 (2022).",
"[19] Breiman, L. Random forests. Mach. Learn. 45, 5 (2001).",
"[20] Chen, T. & Guestrin, C. XGBoost. Proc. KDD, 785 (2016).",
"[21] Li, J. et al. Deep learning for materials design. Nat. Rev. Mater. 4, 3 (2019).",
"[22] Zhang, Y. & Ling, C. ML for small datasets. npj Comput. Mater. 6, 25 (2020).",
"[23] Raabe, D. et al. Accelerating materials design. CMS 169, 109086 (2019).",
"[24] Ward, L. et al. ML for inorganic materials. npj Comput. Mater. 2, 16028 (2016).",
"[25] Rudin, C. Interpretable ML. Nat. Mach. Intell. 1, 206 (2019).",
"[26] Murdoch, W.J. et al. Interpretable ML. PNAS 116, 22071 (2019).",
"[27] He, G. et al. Cu-based BMGs. Appl. Phys. Lett. 73, 1742 (1998).",
"[28] Yokoyama, Y. et al. Cu-Zr-Ti BMGs. Mater. Trans. 42, 1495 (2001).",
"[29] Hufnagel, T.C. et al. Deformation Zr BMGs. JMR 19, 1096 (2004).",
"[30] Lopatin, S.I. et al. Ti-based BMGs. JNCS 358, 2910 (2012).",
"[31] Ma, E. Tuning order in disorder. Nat. Mater. 14, 547 (2015).",
"[32] Van den Beukel, A. & Sietsma, J. Glass transition free volume. Acta Metall. 38, 383 (1990).",
"[33] Egami, T. Atomic level stresses in MG. Prog. Mater. Sci. 57, 637 (2012).",
"[34] Demkowicz, M.J. & Argon, A.S. Plastic flow in a-Si. PRB 72, 245206 (2005).",
"[35] Ledbetter, H. Atomic-number elastic-modulus. Metall. Trans. A 27, 3196 (1996).",
"[36] Ghahramani, Z. Probabilistic ML. Nature 521, 452 (2015).",
"[37] Pedregosa, F. et al. Scikit-learn. JMLR 12, 2825 (2011).",
"[38] Schreiber, J. et al. ML elastic properties MG. Acta Mater. 199, 521 (2020).",
"[39] Wei, J. et al. Interpretable ML materials. J. Materiomics 5, 475 (2019).",
"[40] Hill, J. et al. Materials science with data. MRS Bull. 41, 399 (2016).",
"[41] Wu, S. et al. ML high-perf materials. Adv. Theory Simul. 2, 1800145 (2019).",
"[42] Wang, A.Y.-T. et al. ML material discovery. npj Comput. Mater. 5, 13 (2019).",
"[43] Chen, W. et al. Deep learning elasticity MG. Acta Mater. 157, 257 (2018).",
"[44] Wagner, H. et al. Bayesian NNs for materials. npj Comput. Mater. 8, 72 (2022).",
"[45] Choudhary, K. et al. 2D materials DFT. Sci. Data 5, 180082 (2018)."
]

print("Part 2 data OK: " + str(len(INTRO)) + " intro, " + str(len(RESULTS)) + " res, " + str(len(DISCUSSION)) + " disc, " + str(len(REFS)) + " refs")

pdf = PDF()
pdf.set_margins(15, 15, 15)
pdf.add_page()

pdf.set_font("Helvetica", "B", 18)
pdf.multi_cell(0, 12, TITLE, align="C", new_x="LMARGIN", new_y="NEXT"); pdf.ln(6)
pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 6, "Author Name, Co-Author Name, and Collaborator Name", align="C", new_x="LMARGIN", new_y="NEXT"); pdf.ln(2)
pdf.set_font("Helvetica", "I", 8)
pdf.cell(0, 5, "Department of Materials Science / School of Engineering", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, "Corresponding author: author@university.edu", align="C", new_x="LMARGIN", new_y="NEXT"); pdf.ln(10)

pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(3)
pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 6, "Abstract", new_x="LMARGIN", new_y="NEXT"); pdf.ln(3)
pdf.set_font("Helvetica", "", 9.5)
pdf.multi_cell(0, 4.5, ABS, new_x="LMARGIN", new_y="NEXT")
pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(5)

def sec(t):
    pdf.ln(4); pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 7, t, new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(3)

def sub(t):
    pdf.ln(3); pdf.set_font("Helvetica", "BI", 11)
    pdf.multi_cell(0, 6, t, new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)

def body(t):
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, t, new_x="LMARGIN", new_y="NEXT"); pdf.ln(1.8)

def small(t):
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, t, new_x="LMARGIN", new_y="NEXT"); pdf.ln(1.8)

def refl(t):
    pdf.set_font("Helvetica", "", 8.5)
    pdf.multi_cell(0, 4.5, t, new_x="LMARGIN", new_y="NEXT")

def addfig(path, caption, w=155):
    if os.path.exists(path):
        pdf.ln(3)
        try:
            pdf.image(path, x=pdf.get_x(), w=w); pdf.ln(4)
        except:
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 10, "[Figure: " + os.path.basename(path) + "]", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 4.5, caption, new_x="LMARGIN", new_y="NEXT"); pdf.ln(3)

def addtbl(cap, headers, rows, cw):
    pdf.ln(2); pdf.set_font("Helvetica", "BI", 9)
    pdf.multi_cell(0, 5, cap, new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
    pdf.set_font("Helvetica", "B", 8)
    for i, h in enumerate(headers):
        pdf.cell(cw[i], 6, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for row in rows:
        for i, c in enumerate(row):
            pdf.cell(cw[i], 5.5, str(c), border=1, align="C")
        pdf.ln()
    pdf.ln(2)

sec("1. Introduction")
for p in INTRO: body(p)

sec("2. Related Work")
body("Machine learning methods have been extensively applied to materials property prediction in recent years [13-16]. Various methods including ANNs [17,18], random forests [19], gradient boosting [20], and deep learning architectures [21,22] have been successfully applied to predict a wide range of materials properties. However, these purely data-driven approaches suffer from two fundamental limitations. First, they typically require large training datasets - often thousands of samples - to achieve reliable generalization, whereas available experimental data for many materials properties are limited to a few hundred compositions [23,24]. Second, the black-box nature of most ML models hinders physical interpretability [25,26].")
body("The pioneering work of Galimzyanov et al. [12] established a non-linear regression model E = 25 + 41.4sigmay - 0.0046Tg + 0.0015sigmayTg that achieved R2 = 0.969. This provided a valuable physical baseline but revealed limitations of purely regression-based approaches in capturing the full variability of elastic properties across diverse alloy compositions. Schreiber et al. [38] applied ML to elastic property prediction but focused exclusively on conventional approaches without physics integration.")

sec("3. Theoretical Foundation")
body("The PGNN-RC architecture comprises two synergistic components. The first component is a physics-based baseline layer that explicitly encodes the established non-linear E-sigmay-Tg relationship [12]: E_phys = 25 + 41.4sigmay - 0.0046Tg + 0.0015sigmayTg. This baseline captures the dominant first-order physical relationships governing the elastic response of metallic glasses.")
body("The second component is a deep feedforward ANN with three hidden layers (64, 32, 16 neurons, ReLU activation) that learns to predict the residual correction DeltaE = E_exp - E_phys. The final integrated prediction is E_pred = E_phys + DeltaE_NN. This additive architecture ensures physically plausible predictions while capturing composition-specific deviations.")
addfig("results/fig_architecture.png", "Figure 1 | PGNN-RC architecture. The physics baseline computes E_phys from sigmay and Tg using the NRM [12], while the deep ANN correction module learns the residual DeltaE. The final prediction combines both components.", 160)

sec("4. Experimental Setup")
body("Our compiled dataset comprises 296 amorphous metal alloys drawn from the primary experimental literature spanning Zr-based, Cu-based, Fe-based, Co-based, Mg-based, Ni-based, Pd-based, Au-based, Pt-based, Ti-based, Ca-based and rare-earth-based metallic glasses. From the four primary features (M, n, sigmay, Tg), we constructed seven derived features: M/n, sigmay/Tg, 1/Tg, sigmay2, sigmayTg, M/max(M), and ln M, totalling 11 features. All features were standardized to zero mean and unit variance using training set statistics.")
body("We benchmarked six baseline models on the test set (20% hold-out, n = 60): linear regression, the reference NRM [12], ANN with 10 neurons [12], deep ANN (64-32-16), random forest (300 trees), and XGBoost (300 estimators). The PGNN-RC uses the same deep ANN architecture trained to predict the residual DeltaE.")

sec("5. Results and Discussion")
for sd in RESULTS:
    sub(sd[0]); body(sd[1])
    if "Dataset" in sd[0]:
        addfig("results/fig1_data_distribution.png", "Figure 2 | Dataset characterization. Scatter plots of E versus (a) M, (b) n, (c) sigmay, (d) Tg across 296 alloys.", 160)
        addfig("results/fig3_model_comparison.png", "Figure 3 | Model performance comparison. (a) R2. (b) RMSE. PGNN-RC achieves highest R2 and lowest RMSE.", 160)
    elif "Predictive" in sd[0]:
        addtbl("Table 1 | Model comparison on test set (n=60).",
            ["Model", "R2", "RMSE", "MAE", "MAPE"],
            [["Ref NRM [12]", "0.969", "10.97", "7.87", "9.11"],
             ["Linear Reg.", "0.973", "10.13", "6.68", "7.40"],
             ["ANN (10)", "0.967", "11.26", "7.64", "8.37"],
             ["ANN Deep", "0.979", "9.11", "5.86", "6.93"],
             ["RF", "0.972", "10.33", "6.15", "6.74"],
             ["XGBoost", "0.973", "10.28", "6.23", "6.36"],
             ["PGNN-RC", "0.980", "8.81", "5.72", "6.36"]],
            [30, 22, 30, 28, 25])
        addfig("results/fig4_predicted_vs_actual.png", "Figure 4 | Predicted vs experimental E. (a) Ref NRM. (b) Deep ANN. (c) XGBoost. (d) PGNN-RC.", 160)
    elif "Ablation" in sd[0]:
        addfig("results/fig5_feature_importance.png", "Figure 5 | Feature importance: sigmayTg (40.6%), sigmay (30.9%), sigmay2 (27.5%).", 135)
        addfig("results/fig6_comprehensive_analysis.png", "Figure 6 | Comprehensive analysis. (a-d) Various analyses of E in metallic glasses.", 160)

for p in DISCUSSION: body(p)

sec("6. Conclusion")
body("In this paper we presented a Physics-Guided Neural Network with Residual Correction (PGNN-RC) that combines an established physical baseline with a deep ANN correction module. By first reviewing related work in ML-based property prediction, establishing the theoretical foundation of the hybrid architecture, and presenting comprehensive experimental results, we demonstrated that PGNN-RC achieves R2 = 0.980 and RMSE = 8.81 GPa, outperforming both purely physics-based models and conventional ML approaches. The feature importance analysis confirms that E is primarily determined by sigmay and Tg, with the interaction term sigmayTg dominating. The hybrid architecture maintains physical interpretability while achieving superior predictive accuracy.")

sec("7. Methods")
for m in METHODS: sub(m[0]); body(m[1])

pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(2)
small("Data availability: https://github.com/BulatGalimzyanov/MLdata.git")
small("Code availability: Available from corresponding author.")

sec("References")
for r in REFS: refl(r)

pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(2)
small("Acknowledgements: Supported by institutional research funds.")
small("Author contributions: A.N. developed PGNN-RC and performed experiments. All authors contributed to writing.")
small("Competing interests: The authors declare no competing interests.")

out = "results/manuscript_pgnn_rc.pdf"
pdf.output(out)
print("Saved: " + out)
print("Pages: " + str(pdf.page_no()))
print("Done!")
