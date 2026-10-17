# Submission form answers (paste-ready)

## Project Description

Robot Agent Lab is a team of NVIDIA Nemotron agents that runs a science lab's night shift on a simulated SO-101 robot arm. Sample tubes arrive all day with coloured caps; before the night runs, red caps must go to the cold rack, blue caps to the sequencer rack, and green controls must stay on the bench. A tube in the wrong rack ruins an expensive run. Today a technician does this twice a night. The human types one line, "Put tonight's samples where they belong.", and walks away.

Five agents share one loop on Crusoe Managed Inference. A planner (Nemotron 3 Super) turns the goal into a definition of done that can be checked from a photo or robot state. An operator (Nemotron 3.5 Lightning) drives the arm through six tools and must cite evidence for every check before claiming done. A sealed verifier (Nemotron Nano Omni) sees only the checks and one top-down photo, never the operator, and votes three times. A rule-based monitor catches loops, stalls and collapsing progress and decides whether to continue, re-plan, stop or ping a human. A curator writes each shift into a shared brain: a ledger of facts with validity windows and provenance, where a new value for the same fact supersedes the old one and nothing is deleted. An improver rewrites the team's own playbook and keeps the change only if it scores at least as well on a shift the team has never seen.

The bench has injected quirks the memory has to learn: a gripper that ignores the first close, a rack that moves at shift 3, a camera nudged at shift 5. Because it is a simulation, we have ground truth, so we report the verifier's own accuracy, not just the agents'. In the recorded runs the memory superseded the moved rack with zero stale positions served, every false "done" claim was caught by the verifier, and the first playbook rewrite was kept because it removed a false completion on a held-out shift. The caveat is in the README: memory has not yet raised the success rate or reduced robot actions.

## Public GitHub Repository

https://github.com/abhijitbetigeri/robot-agent-lab

## Public Demo Video

Direct MP4 (public, plays in the browser): https://abhijitbetigeri.github.io/robot-agent-lab/assets/demo.mp4

Landing page with the video embedded: https://abhijitbetigeri.github.io/robot-agent-lab/

If the form insists on YouTube: upload demo/robot-agent-lab-demo.mp4 as Unlisted and paste that link instead.

## Which problem statement

Shared Brain

## NVIDIA Best Use of Nemotron

Yes

## NVIDIA Best Use of NemoClaw & OpenShell

No. The operator's OpenShell sandbox policy is written (nemoclaw/policy.yaml) but we did not get NemoClaw running on the day (it needs Node 22.19 and the machine had Node 20), so we are not claiming it.

## Did you use NVIDIA tools

Yes (Nemotron 3 Super, Nemotron 3.5 Lightning, Nemotron 3 Nano, Nemotron 3 Nano Omni)

## Did you use You.com tools

Yes (You.com Search API as the operator's `research` tool, agents/researcher.py)

## Describe your use of Crusoe Managed Inference

Every model call in the project goes through Crusoe's OpenAI-compatible endpoint. We mapped five agent roles to four Nemotron models (Super for planning, Lightning for the operator, Nano for the monitor and curator, Nano Omni for the image verifier) so rate limits spread across models. We used tool calling, chat_template_kwargs for the thinking toggle, base64 image input for the verifier, exponential backoff on 429 and 503, and GET /v1/models to confirm exact model IDs before the first run. Over the day we sent roughly 350 requests and about 1.5 million prompt tokens for the eval sweeps, the self-improvement cycle and the live demo shifts. Per-role token usage is logged for every shift and feeds the results table.

## Feedback on Crusoe products

What worked: the hackathon skill file was the single most useful document of the day; it saved us from the deprecated-model and key-quoting mistakes before we made them. Tool calling on Nemotron 3.5 Lightning was reliable, image input to Nano Omni worked first time, and we never hit a rate limit with a card on file.

What cost us time: four of our six model IDs were spelled differently in the docs than in the API (the NVIDIA- prefix on Super and Nano, the casing on Lightning); the /v1/models check caught it, but the docs table should match the API. The usage object never returned prompt_tokens_details for the Nemotron models, so we could not measure prefix-cache savings at all, which we had planned to report. The features list in /v1/models came back empty for every model, so we could not programmatically tell which models support tools. Lightning emits one tool call per turn, which makes agent budgets long; a note about that in the model guidance would help.

## Feedback on partner products

Nemotron: Super wrote good definitions of done on the first try. Lightning was a dependable operator but turned a "progress line" instruction into a nonexistent tool call until we made progress reporting a real tool. Nano Omni as a sealed verifier was right 75 percent of the time on simple scenes; it once judged a tube 8 cm from a patch as inside it, and only one of three samples returned parseable JSON on some calls. Nano as a curator wrote verbose, low-value facts until we capped it and added deterministic auto-facts; a smaller model needs tighter write schemas.

NemoClaw and OpenShell: the install requires Node 22.19 or newer; our machine had Node 20 and the day did not leave time to upgrade, so we shipped the policy file without running it. A Node-version check up front in the installer would save people the attempt.

You.com: the Search API was simple to call and the results shape (results.web with snippets, title, url) was easy to turn into cited facts with a retrieval date. Having the response shape in the quickstart would have saved one round trip.
