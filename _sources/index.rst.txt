Open Kinetics Predictor
=======================

.. raw:: html

   <section class="hero-panel">
     <div class="hero-copy">
       <p class="eyebrow">Enzyme Kinetics Prediction Platform</p>
       <h1>Predict enzyme kinetic parameters from protein sequence and substrate data.</h1>
       <p class="lead">
         Open Kinetics Predictor unifies published machine learning models behind
         one asynchronous web and API workflow for kcat, Km, and kcat/Km
         prediction.
       </p>
     </div>
     <div class="hero-stat-grid">
       <div class="hero-stat">
         <span class="hero-stat-label">Inputs</span>
         <strong>Protein sequences and molecule strings</strong>
       </div>
       <div class="hero-stat">
         <span class="hero-stat-label">Outputs</span>
         <strong>Structured kinetic predictions</strong>
       </div>
       <div class="hero-stat">
         <span class="hero-stat-label">Workflow</span>
         <strong>Validate, submit, track, download</strong>
       </div>
     </div>
   </section>

.. raw:: html

   <section class="feature-band">
     <article class="feature-card">
       <p class="feature-kicker">Submit</p>
       <h2>Run batch prediction jobs</h2>
       <p>Validate tabular protein and substrate inputs, submit asynchronous jobs, and collect result files after completion.</p>
     </article>
     <article class="feature-card">
       <p class="feature-kicker">Compare</p>
       <h2>Use published engines</h2>
       <p>Access KinForm, UniKP, DLKcat, TurNup, EITLEM, CataPro, CatPred, OmniESI, RealKcat, IECata, and MMISA-KM through one interface.</p>
     </article>
     <article class="feature-card">
       <p class="feature-kicker">Extend</p>
       <h2>Add new methods</h2>
       <p>Follow the contributor guides for descriptors, runtime paths, embedding cache reuse, similarity datasets, and observability conventions.</p>
     </article>
   </section>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Project

   project/overview
   project/input_data
   project/supported_methods
   project/prediction_workflow
   project/api_and_limits

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Contributor Guides

   project/contributing
   project/plm_embedding_cache
   project/mmseqs_similarity_datasets
   project/observability

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference

   api_reference
   README
