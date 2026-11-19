# fly-memory

using a fly's brain for ai

taking the actual neural connections of a fruit fly and seeing if we can use them to build an ai that learns, remembers, and adapts.

the idea is to test whether biological neural networks can learn more efficiently, retain memories better, and generalize to new experiences compared to regular neural networks.

starting with the flywire connectome (139k neurons, 2.7m connections), adding learning and memory, and comparing it against randomly wired networks.

basically, trying to figure out if there's something about how brains are wired that makes them better at learning.

## experiments

### 1. can it learn an association?

trained a simple synaptic model to respond to one synthetic odor, using real and shuffled wiring.

**result:** both learned. the real response went from 0.0429 to 0.1113; shuffled went from 0.0539 to 0.1397.

**takeaway:** the update rule guaranteed an increase. not evidence that real wiring is better.

### 2. does real wiring learn and remember better?

trained on 20 odors in sequence, tested altered versions, and compared retention across 30 seeds.

**result:** real wiring learned at 69.6% and retained 69.7%; shuffled learned at 81.7% and retained 79.7%.

**takeaway:** shuffled won this benchmark. our task and controls are still pretty artificial.

### 3. can a forgotten memory come back?

added active and silent traces, reset the active one, then tried recovering it from partial reminders. tested wrong cues, erased traces, and synapse lesions.

**result:** real recovered to 76.3% from 50%; shuffled reached 83.3%. wrong cues stayed near chance and erasing the silent trace stopped recovery.

**takeaway:** the recovery mechanism works, but we programmed the forgetting and recovery rule ourselves. we haven't discovered biological recovery yet.

### 4. can memories survive interference? (next)

learn associations, introduce conflicting experiences, then see if partial cues can recover old memories without retraining. compare real wiring, shuffled wiring, and targeted circuit lesions.

**status:** not run yet.
