# Web-MCPT Diagnostic Record

**Project:** `stan_new_diagnostic`\
**Scope:** Web-MCPT formulation and implementation audit\
**Historical reference:** `stan_new/` is kept unchanged.
`stan_new_diagnostic/` was created as a copy of `stan_new/`, so
diagnostic variants do not alter the historical reproduction.

> **Status:** Web-MCPT audit substantially completed. Model formulation,
> priors/transformations, numerical safeguards, scaling, and
> reference-point construction have been checked. Utility clipping can
> be removed. The retained posterior does not appear to depend on the
> existing hard parameter caps, but the final cap policy is deferred
> until the Lab-ET audit.

## 1. Purpose

This audit was conducted before the target-domain composition
sensitivity analysis to distinguish: (1) the substantive MCPT
formulation inherited from the original implementation; (2) Bayesian
parameterization choices; (3) numerical safeguards such as caps and
utility clipping; and (4) implementation details that could affect
prediction, PSIS-LOO, and stacking.

The objective is not to redesign MCPT. The paper remains focused on
model fusion/stacking.

## 2. Web-MCPT formulation

The audited Stan model uses

\[ ASC=`\eta`{=tex}*{ASC},`\quad`{=tex}
`\alpha`{=tex}=`\operatorname{logit}`{=tex}\^{-1}(`\eta`{=tex}*`\alpha`{=tex}),`\quad`{=tex}
`\lambda=1`{=tex}+`\exp`{=tex}(`\eta`{=tex}\_`\lambda`{=tex}),`\quad`{=tex}
r=`\exp`{=tex}(`\eta`{=tex}\_r),`\quad`{=tex}
`\beta`{=tex}*k=`\exp`{=tex}(`\eta`{=tex}*{`\beta`{=tex}\_k}), \]

subject to numerical caps in the historical Stan implementation. The Web
beta vector is \[
\[`\beta`{=tex}*{RC},`\beta`{=tex}*{OC},`\beta`{=tex}*{OC},`\beta`{=tex}*{DR}\],
\] so the two uncertain charging-cost states share
(`\beta`{=tex}\_{OC}).

For (z=(x-`\tilde{x}`{=tex})`\beta`{=tex}),

\[ v(z)=
```{=tex}
\begin{cases}
z^\alpha,&z\ge0,\\
-\lambda(-z)^\alpha,&z<0.
\end{cases}
```
\]

The Prelec function is

\[ `\pi`{=tex}(p)=`\exp[-(-\ln p)^r]`{=tex}. \]

For the two EV charging-cost states,

\[ W_1=`\pi`{=tex}(P_1),`\qquad `{=tex}W_2=1-`\pi`{=tex}(P_1), \]

equivalent to (W_2=`\pi`{=tex}(P_1+P_2)-`\pi`{=tex}(P_1)) because
(P_1+P_2=1).

The utility structure corresponds to

\[ U\_{CV}=V\_{CV,RC}+V\_{CV,OC}+V\_{CV,DR}, \]

\[ U\_{EV}=V\_{EV,RC}+W_1V\_{EV,OC}+W_2V\_{EV,OC_1}+V\_{EV,DR}+ASC. \]

**Decision:** Pass. No formulation change is currently required for the
value function, Prelec weighting, ASC placement, or shared OC
coefficient.

## 3. Bayesian priors

Current priors on unconstrained parameters are:

\[ `\eta`{=tex}*{ASC}`\sim `{=tex}N(0,1),`\quad`{=tex}
`\eta`{=tex}*`\alpha`{=tex}`\sim `{=tex}N(0,0.75),`\quad`{=tex}
`\eta`{=tex}\_`\lambda`{=tex}`\sim `{=tex}N(0,0.3),`\quad`{=tex}
`\eta`{=tex}*r`\sim `{=tex}N(0,0.2),`\quad`{=tex}
`\eta`{=tex}*{`\beta`{=tex}\_k}`\sim `{=tex}N(0,0.3). \]

A 100,000-draw transformation sanity check gave approximate central 95%
intervals:

  Parameter     Approx. central 95% interval
  ----------- ------------------------------
  ASC                        \[-1.97, 1.97\]
  alpha                       \[0.19, 0.81\]
  lambda                      \[1.56, 2.80\]
  r                           \[0.67, 1.48\]
  beta                        \[0.56, 1.80\]

Interpretation: (`\alpha`{=tex}`\in`{=tex}(0,1)) gives diminishing
sensitivity; (`\lambda`{=tex}\>1) loss aversion; (r\>0) is valid for the
Prelec function and (r=1) gives identity weighting; positive
(`\beta`{=tex}) preserves the sign of the reference deviation.

**Decision:** No prior redesign at this stage. Keep this as diagnostic
documentation rather than a focus of the stacking paper.

## 4. Hard parameter caps

### 4.1 Historical hard caps

The historical Stan implementation contains the following numerical safeguards:

\[
\lambda=1+\min(\exp(\eta_\lambda),20),
\]

\[
r=\min(\exp(\eta_r),3),
\]

\[
\beta_k=\min(\exp(\eta_{\beta_k}),20).
\]

The corresponding effective upper limits are therefore approximately:

- \(\lambda\leq21\);
- \(r\leq3\);
- \(\beta_k\leq20\).

Because these caps alter the parameter transformations, the first diagnostic question was whether the posterior under the existing model showed evidence of approaching or accumulating at these boundaries.

### 4.2 Boundary analysis of the existing posterior

Before changing the model, the retained posterior draws from the existing capped specification were inspected across the 10 audited data-split seeds.

The analysis showed:

- 0 divergences across the audited runs;
- no evidence of posterior draws accumulating at the imposed caps;
- approximate observed posterior maxima of:
  - \(\beta_{RC}\): 3.41;
  - \(\beta_{OC}\): 6.53;
  - \(\beta_{DR}\): 3.92;
  - \(\lambda\): 9.29;
  - \(r\): 2.478.

Relative to the imposed boundaries, the beta parameters and \(\lambda\) remained far below their caps. The posterior maximum of \(r\) was comparatively closer to its upper bound of 3, but there was no observed accumulation at the boundary.

**Intermediate conclusion:** The existing retained posterior did not provide evidence that the hard caps were actively truncating posterior mass. However, a boundary check alone cannot establish what would happen if the caps were removed, particularly during warm-up and exploration of extreme proposals. This motivated a controlled no-caps re-estimation.

### 4.3 Controlled removal of hard caps

A diagnostic variant removed the hard parameter caps (together with utility clipping in the corresponding combined diagnostic) and re-estimated the model.

For seed 12, two NaN proposal warnings occurred at the first warm-up iteration (chains 4 and 3). Sampling subsequently completed successfully. The retained posterior showed:

- 0 divergences;
- good \(\hat R\) and ESS diagnostics;
- posterior means within approximately 1% of the capped reference run;
- upper posterior quantiles and maxima remaining well below the historical caps.

These results are consistent with the earlier boundary analysis: the retained posterior inference does not appear to rely on the imposed caps. At the same time, the initial NaN proposal warnings suggest that the caps may have served as protection against extreme proposals during early warm-up.

### 4.4 Decision

The two diagnostics provide complementary evidence:

1. **Boundary analysis under the existing capped model:** no evidence of retained posterior accumulation at the caps.
2. **Controlled no-caps re-estimation:** removing the caps produces essentially unchanged retained posterior inference, although extreme early warm-up proposals can occur.

Therefore, there is currently no evidence that the hard caps materially determine the retained Web-MCPT posterior. However, the final decision on whether to remove or retain them is **deferred until the Lab-ET audit**, so that numerical safeguards can be assessed consistently across the two component models.

## 5. Utility clipping

The historical Stan implementation clips utilities to approximately
(\[-20,20\]) in the model and generated quantities.

Post-hoc diagnostics found:

-   no utilities above +20;
-   rare lower-tail clipping;
-   minimum CV utility about -29.20;
-   minimum EV utility about -27.33;
-   the worst observed draw-by-task clipping share was very small (about
    0.0411% for CV in the most affected diagnostic).

A controlled no-clipping variant, retaining parameter caps, had 0
divergences, good sampling diagnostics, and negligible changes in
posterior estimates, prediction, and LOO quantities.

**Decision:** Remove utility clipping. Stable probability calculation
should rely on max-centering/softmax rather than modifying the utility
function.

## 6. Combined no-caps/no-clipping check

Removing both caps and clipping left retained posterior inference
essentially unchanged after the initial warm-up proposal warnings
described above.

**Decision:** Utility clipping = remove. Hard caps = defer final
decision until Lab-ET audit.

## 7. Scaling

Preprocessing uses

\[ RC\^*=-`\operatorname{MinMax}`{=tex}(RC),`\quad`{=tex}
OC\^*=-`\operatorname{MinMax}`{=tex}(OC),`\quad`{=tex}
OC_1\^\*=-`\operatorname{MinMax}`{=tex}(OC_1), \]

\[ DR\^\*=`\operatorname{MinMax}`{=tex}(`\log `{=tex}DR). \]

This is consistent with the original MCPT implementation. In the
split-based workflow, scaler parameters are fitted on training data and
applied to the corresponding test data, avoiding test-information
leakage.

**Decision:** Pass. No current evidence of an implementation error in
scaling.

## 8. Reference-point construction

Diagnostic script:

`notes/oriniginal_mcpt_referencepoint_check.py`

For the first inspected Lab task (ID 11165, Scenario R1.5s), raw values
were:

  Alternative       RC        OC    DR    P1    P2   P1_V   P2_V
  ------------- ------ --------- ----- ----- ----- ------ ------
  CV              1800        55   500   ---   ---    ---    ---
  EV              2350   missing   300   0.9   0.1     18     22

After preprocessing/scaling:

\[ X\_{CV}=\[-0.12903226,-0.5,-0.46666667,0.46497352\], \]

\[ X\_{EV}=\[-0.24731183,-0.11458333,-0.1,0\]. \]

The original implementation uses

\[ `\tilde{x}`{=tex}*{RC}=X*{CV,RC},`\qquad`{=tex}
`\tilde{x}`{=tex}*{DR}=X*{CV,DR}, \]

while charging cost uses

\[ `\tilde{x}`{=tex}*{OC}=`\tilde{x}`{=tex}*{OC_1}
=-`\left`{=tex}(P_1X\_{EV,OC}+P_2X\_{EV,OC_1}`\right`{=tex}). \]

For this task,

\[ -`\left[0.9(-0.11458333)+0.1(-0.1)\right]=0.113125`{=tex}. \]

Thus

\[ `\tilde{x}`{=tex}=\[-0.12903226,0.113125,0.113125,0.46497352\]. \]

**Finding:** RC and DR use task-specific CV-based references. The
uncertain charging-cost reference is constructed from the
probability-weighted EV charging-cost states with the sign convention
used in the original code. This behavior already exists in the original
MCPT implementation; it was not introduced by the Stan restructuring.

**Decision:** Document but do not redesign now. In the paper, describe
the reference-point construction generally and attribute it to the
original MCPT implementation rather than stating that every attribute
directly uses the CV value as its reference.

## 9. Model vs generated quantities

The likelihood and generated-quantities calculations were checked for
consistency. Generated quantities use stable max-centering/softmax and a
small probability floor for log-probability calculation. If utility
clipping is removed from the final model, it should be removed
consistently from both the model and generated-quantities calculations.

Unused numerical helper code should be cleaned only after the final
specification is fixed.

## 10. Decision summary

  -----------------------------------------------------------------------
  Audit item              Status                  Decision
  ----------------------- ----------------------- -----------------------
  MCPT value function     Pass                    Retain

  Prelec weighting        Pass                    Retain

  ASC structure           Pass                    Retain

  Positive beta           Checked                 Retain pending
  transformation                                  cross-model alignment

  Bayesian priors         Checked                 No redesign

  Scaling                 Pass                    Retain

  Reference-point         Documented inherited    Do not redesign now
  construction            issue                   

  Utility clipping        Unnecessary             Remove

  Hard parameter caps     Not active in retained  Final decision deferred
                          posterior               

  Stable probability      Pass                    Retain
  calculation                                     

  Web/Lab alignment       Pending                 Resolve after Lab-ET
                                                  audit
  -----------------------------------------------------------------------

## 11. Diagnostic artifacts and traceability

  -------------------------------------------------------------------------------------------------
  Artifact                                          Purpose                 Status
  ------------------------------------------------- ----------------------- -----------------------
  `notes/audit_log.csv`                             Master issue/checklist  Keep
                                                    index                   

  `notes/web_mcpt_diagnostics.md`                   Detailed                Keep
                                                    reasoning/decision      
                                                    record                  

  `notes/oriniginal_mcpt_referencepoint_check.py`   Raw → scaled X →        Keep
                                                    reference → X-reference 
                                                    check using original    
                                                    MCPT logic              

  Historical Web Stan script                        Reference specification Keep historical copy
                                                                            unchanged

  No-utility-clipping variant                       Controlled clipping     Keep as diagnostic
                                                    ablation                

  No-caps/no-clipping variant                       Numerical-safeguard     Keep as diagnostic
                                                    ablation                

  Posterior cap-summary outputs                     Check proximity to caps Keep

  Utility-clipping diagnostic outputs               Quantify clipping       Keep

  Prior-transformation diagnostic output            Inspect implied         Keep
                                                    transformed priors      

  10-seed comparison outputs                        Check robustness        Keep
  -------------------------------------------------------------------------------------------------

Exact local filenames not explicitly established in the audit record
should be copied from the local directory rather than reconstructed from
memory.

For every diagnostic variant, preserve four pieces of information: why
it was created; what differs from the reference model; what result it
produced; and whether the change entered the final specification.

## 12. Current stopping point

The Web-MCPT audit can be treated as substantially complete before
moving to `lab_ET_mcpt_stan.py`.

The main unresolved cross-model decision is the hard-cap policy. Revisit
it after checking the Lab model's (`\beta`{=tex}\_{k0}) transformation,
attention-dependent softplus mapping, (`\gamma`{=tex}\_k)
parameterization/priors, numerical behavior, and consistency with the
Web formulation. Only after that comparison should the final Web and Lab
Stan files be cleaned and frozen for the target-domain composition
sensitivity analysis.
