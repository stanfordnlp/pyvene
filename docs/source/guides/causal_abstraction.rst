A Little Guide to Causal Abstraction
====================================

*From Interventions to Gaining Interpretability Insights*

:author: Zhengxuan Wu

Basic interventions are fun but we cannot make any causal claim
systematically. To gain actual interpretability insights, we want to
measure the counterfactual behaviors of a model in a data-driven
fashion. In other words, if the model responds systematically to your
interventions, then you start to associate certain regions in the
network with a high-level concept. We also call this alignment search
process with model internals.

Understanding Causal Mechanisms with Static Interventions
---------------------------------------------------------

Here is a more concrete example,

.. code:: python

   def add_three_numbers(a, b, c):
       var_x = a + b
       return var_x + c

The function solves a 3-digit sum problem. Let's say, we trained a
neural network to solve this problem perfectly. "Can we find the
representation of (a + b) in the neural network?". We can use this
library to answer this question. Specifically, we can do the following,

-  **Step 1:** Form Interpretability (Alignment) Hypothesis: We
   hypothesize that a set of neurons N aligns with (a + b).
-  **Step 2:** Counterfactual Testings: If our hypothesis is correct,
   then swapping neurons N between examples would give us expected
   counterfactual behaviors. For instance, the values of N for (1+2)+3,
   when swapping with N for (2+3)+4, the output should be (2+3)+3 or
   (1+2)+4 depending on the direction of the swap.
-  **Step 3:** Reject Sampling of Hypothesis: Running tests multiple
   times and aggregating statistics in terms of counterfactual behavior
   matching. Proposing a new hypothesis based on the results.

To translate the above steps into API calls with the library, it will be
a single call,

.. code:: python

   intervenable.eval_alignment(
       train_dataloader=test_dataloader,
       compute_metrics=compute_metrics,
       inputs_collator=inputs_collator
   )

where you provide testing data (basically interventional data and the
counterfactual behavior you are looking for) along with your metrics
functions. The library will try to evaluate the alignment with the
intervention you specified in the config.

Understanding Causal Mechanism with Trainable Interventions
-----------------------------------------------------------

The alignment searching process outlined above can be tedious when your
neural network is large. For a single hypothesized alignment, you
basically need to set up different intervention configs targeting
different layers and positions to verify your hypothesis. Instead of
doing this brute-force search process, you can turn it into an
optimization problem which also has other benefits such as distributed
alignments.

In its crux, we basically want to train an intervention to have our
desired counterfactual behaviors in mind. And if we can indeed train
such interventions, we claim that causally informative information
should live in the intervening representations! Below, we show one type
of trainable intervention :class:`RotatedSpaceIntervention <pyvene.models.interventions.RotatedSpaceIntervention>`
as,

.. code:: python

   class RotatedSpaceIntervention(TrainableIntervention):
       
       """Intervention in the rotated space."""
       def forward(self, base, source):
           rotated_base = self.rotate_layer(base)
           rotated_source = self.rotate_layer(source)
           # interchange
           rotated_base[:self.interchange_dim] = rotated_source[:self.interchange_dim]
           # inverse base
           output = torch.matmul(rotated_base, self.rotate_layer.weight.T)
           return output

Instead of activation swapping in the original representation space, we
first **rotate** them, and then do the swap followed by un-rotating the
intervened representation. Additionally, we try to use SGD to **learn a
rotation** that lets us produce expected counterfactual behavior. If we
can find such rotation, we claim there is an alignment.
``If the cost is between X and Y.ipynb`` tutorial covers this with an
advanced version of distributed alignment search, `Boundless
DAS <https://arxiv.org/abs/2305.08809>`__. There are `recent
works <https://www.lesswrong.com/posts/RFtkRXHebkwxygDe2/an-interpretability-illusion-for-activation-patching-of>`__
outlining potential limitations of doing a distributed alignment search
as well.

You can now also make a single API call to train your intervention,

.. code:: python

   intervenable.train_alignment(
       train_dataloader=train_dataloader,
       compute_loss=compute_loss,
       compute_metrics=compute_metrics,
       inputs_collator=inputs_collator
   )

where you need to pass in a trainable dataset, and your customized loss
and metrics function. The trainable interventions can later be saved on
to your disk. You can also use :class:`intervenable.evaluate() <pyvene.models.intervenable_base.IntervenableModel>` your
interventions in terms of customized objectives.