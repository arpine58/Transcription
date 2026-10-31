# How I worked with the AI assistant

I built this project in a Claude Cowork project with several threads. Claude wrote most of the code and docs.
I ran everything on my own laptop, recorded the test audio, made the decisions and reviewed every change before
committing. This file is the short version; the full conversations are in `chats/`.

## The setup

- **One project, several threads.** Planning, building and documentation each had their own thread, so each
  conversation stayed on one topic.
- **Rules written down once.** `CLAUDE.md` holds the requirements and working rules: keep it simple, fail loudly,
  one module per job, choose models with measured numbers, three failed attempts and we stop. Every thread
  starts from it, so I didn't have to repeat them.
- **The AI never touched git.** Claude wrote changes into my local folder. I read the diff, ran it, and committed
  and pushed myself.
- **A decision log.** Every important choice, with its reason and numbers, went into `DECISIONS.md` the moment it
  was made. That is the record of why the code looks the way it does.

## The process, step by step

1. **Plan before code.** Claude turned the brief into a plan (`PLAN.md`, `docs/TECH_PLAN.md`) and a list of open
   questions. I sent the questions to Mathrix (budget, Zoom, dialect, number of speakers, the 5 s rule,
   separate steps) and we planned against their answers, not guesses.
2. **A thin end-to-end version first.** The first working version captured both audio sides, cut phrases,
   transcribed, translated and printed lines, all with simple choices. Then each part was improved on its own.
3. **Measure instead of guessing.** When the Armenian came out as gibberish, we did not switch models on
   impressions. I recorded 12 test clips and typed exactly what I said. Claude wrote `bench/asr.py` to score each
   model on them (letter and word error rates, speed). Every model change since then is backed by that table.
4. **I made the decisions that change the goal.** For example, when large-v3 was accurate but too slow and medium
   was fast but poor, Claude laid out the options with numbers and I chose "large-v3, tuned for speed". That led
   to the 4-bit model we use now.
5. **Tests from the requirements.** `TESTS.md` lists the checks derived from the brief; the unit tests in
   `tests/` were written from those, not from the code.
6. **Clean up and document at the end.** README, limitations and this file, then a cleanup pass that removed the
   code for dropped experiments.

## Where the AI was wrong, and how we caught it

Running everything on the real laptop was the check. These all happened:

- **First model choice.** Whisper large-v3-turbo, suggested as fast enough, produced English, Tibetan and Latin
  letters instead of Armenian. Caught in the first live test. We added checks that show bad output as `(unclear)`
  and moved to measured model choices.
- **Install instructions that broke.** pip pulled NVIDIA builds of torch that crashed (`libcudart.so`), and a
  suggested version (torchaudio 2.15 for CPU) did not exist. The model export failed on a torchvision mismatch.
  Each was fixed from the actual error, and the working versions are pinned in the README.
- **The GPU was invisible.** OpenVINO only listed the CPU until the Intel GPU driver was installed; that step is
  now in the setup.
- **A promising model that never ran.** The Armenian fine-tune BuzzASR failed three different ways. Following the
  three-attempts rule, we stopped and documented it as the first thing to retry.
- **An unfair test.** The first "Armenian hint" sentence for Whisper was word for word the same as two test clips,
  which made the results look better than they were. It showed up in the per-clip output; we re-measured
  with a neutral sentence, the gain disappeared, and the feature was removed.
- **Small code bugs.** A module-level import broke the speaker tests, and the latency log showed speaker scores on
  my own lines. Both were found by running the tests and reading the log.

## What I'd keep doing

- Ask the client before building on assumptions.
- Get something running end to end on day one.
- Trust numbers measured on my own machine over model cards and the AI's confidence.
- Read the raw output, not only the summary table.
- Keep the AI's changes out of git until I have run and reviewed them.
