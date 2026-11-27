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

### 4. can memories survive interference?

trained two memory pathways on 12 associations, then introduced conflicting experiences and tested whether partial reminders could recover old memories without retraining.

**result:** real wiring went from 99.2% learned to 17.2% after interference, then recovered to 37.5%. shuffled wiring recovered from 12.2% to 31.9%. the extra recovery from real wiring was just +0.6 ± 3.9 percentage points.

**takeaway:** the slow memory pathway helped reduce interference, but real biological wiring didn't show a convincing advantage. recovery was still below chance, and the mechanism was engineered rather than discovered.

### 5. can the same odor hold two conflicting memories?

used real flywire connections to mbon11 and mbon18. trained 12 odors with one set of associations, reversed them in a different context, then tested whether both memories could still be retrieved.

**result:** real wiring retained 85.3% of old associations versus 75.3% for shuffled wiring. new-context accuracy was 91.7% versus 90.6%. real wiring showed a +10.0 ± 5.2 percentage point advantage in old-memory retention.

**takeaway:** our first promising advantage from real biological wiring. still preliminary — need to check whether it comes from meaningful circuit structure or simpler connectivity differences.

### 6. is the biological wiring actually responsible?

tested activity-matched shuffling, other mbon pairs, and selective removal of highly connected neurons.

**result:** real wiring achieved 80.3% joint memory accuracy versus 69.2% for random wiring, but 78.6% for activity-matched wiring. the matched shuffle changed 93.9% of connected kc rows. targeted lesions unexpectedly improved accuracy to 84.7%.

**takeaway:** the original advantage may depend more on activity-connectivity relationships than exact wiring. selective suppression might reduce interference, but we still need strength-matched controls.

### 7. can protecting vulnerable synapses prevent forgetting?

identified kenyon cells supporting old memories that were also active during conflicting learning. protected their synaptic updates and compared targeted, matched, random, and importance-based protection across 50 seeds.

**result:** real wiring achieved 80.8% joint accuracy without protection, 78.5% with targeted protection, and 79.5% with matched protection. importance-based protection improved old memory but significantly hurt new learning.

**takeaway:** selective freezing didn't improve memory. protecting old information alone creates a tradeoff with learning new things. the next question is whether we can identify genuinely conflicting updates instead of simply protecting important connections.

### 8. can a biological circuit learn through experience?

built a recurrent, plastic mushroom body circuit using 2,934 real neurons. placed it in a changing environment where it learns through rewards rather than labels.

**result:** across 153,600 training episodes, the real circuit achieved 85.9% final accuracy and 66.4% retention. matched random wiring and a linear rl agent performed similarly. slow memory helped against decay but didn't outperform persistent weights.

**takeaway:** adding biological complexity isn't enough. our engineered dynamics still limit what the network can do. instead of guessing how the brain should learn, let's use ml to discover how it could learn.
