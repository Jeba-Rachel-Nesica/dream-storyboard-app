#!/usr/bin/env python3
"""
Generate 1,000 unique IRT-based nightmare-comfort pairs.
Based on published Imagery Rehearsal Therapy principles.
"""

import json
import random
from itertools import product, combinations


# IRT Components (from published research)
GROUNDING = [
    "You feel the floor solid beneath you.",
    "You notice your breath, steady and real.",
    "You sense the weight of your body, safe and grounded.",
    "You feel the chair supporting you.",
    "You become aware of the room around you, familiar and secure.",
    "You notice the temperature of the air on your skin.",
    "You feel your feet on the ground, stable and present.",
    "You hear the quiet sounds of safety around you.",
    "You feel the texture of the surface beneath your hands.",
    "You notice the steady rhythm of your heartbeat.",
]

MASTERY = [
    "You press pause on the scene like a remote; everything holds still while you choose your next step.",
    "You turn to the {threat} and say, 'Stop.' It does.",
    "You realize you can rewind this moment and choose differently.",
    "You notice a door you hadn't seen before; the handle turns easily.",
    "You find you can slow down time, giving yourself space to think.",
    "You discover you have a voice in this dream, and you use it.",
    "You realize you can change the lighting, making the space brighter.",
    "You step back and observe the scene as if watching a movie you can edit.",
    "You notice you can control the pace of what happens next.",
    "You find an exit that wasn't there before, and it's open.",
]

THREAT_REDUCTION = [
    "The {threat} shrinks to the size of a pebble; you set it on a shelf.",
    "You turn on a light; the {threat} resolves into something ordinary.",
    "The {threat} fades like morning mist in sunlight.",
    "You look closer and see the {threat} is just a shadow cast by something harmless.",
    "The {threat} transforms into something neutral and unthreatening.",
    "You watch as the {threat} loses its power, becoming small and manageable.",
    "The {threat} dissolves when you face it directly.",
    "The {threat} becomes transparent, and you can see through it.",
]

CLOSURE = [
    "You leave the space knowing you can return on your own terms.",
    "The quiet that follows is friendly and warm.",
    "You step into open light, feeling the warmth on your face.",
    "You notice you're smiling, knowing you handled this.",
    "You walk away with confidence, the fear behind you.",
    "You take a deep breath and feel peace settling in.",
    "You realize you're safe, and this feeling of safety stays with you.",
]

# Nightmare templates (1000+ combinations)
NIGHTMARE_TEMPLATES = {
    "exam_academic": [
        "i arrive at the final exam for a class i never attended",
        "i sit for an exam and the questions are in a language i don't know",
        "the exam starts and i realize i studied the wrong subject",
        "i open my test booklet and all the pages are blank",
        "the teacher announces a surprise exam i didn't prepare for",
        "i'm taking an exam but my pen won't write anything",
        "the clock runs out before i can finish the first question",
        "everyone else is writing but i can't understand the questions",
        "i realize halfway through that this is the wrong exam room",
        "the exam paper keeps changing; questions disappear as i read them",
    ],
    "presentation": [
        "i'm giving a presentation but forgot all my slides",
        "i stand before the class and can't remember my topic",
        "my presentation won't load and important people are waiting",
        "i'm speaking but my words come out as gibberish",
        "the projector shows embarrassing content instead of my slides",
        "i realize mid-presentation that i prepared the wrong material",
        "everyone is staring and i've forgotten what to say",
        "my notes are missing and i'm supposed to present now",
        "the microphone won't work and no one can hear me",
        "i trip on stage and all my materials scatter everywhere",
    ],
    "social_judgment": [
        "i walk into the party and everyone stops talking to stare at me",
        "i'm eating with others and spill food all over myself",
        "everyone is laughing and i realize they're laughing at me",
        "i'm wearing the wrong clothes and everyone notices",
        "i try to join the conversation but say something terrible",
        "i arrive late and everyone has been waiting for me, annoyed",
        "my phone buzzes; it's a group chat making fun of me",
        "i wave at someone who looks away and doesn't acknowledge me",
        "i'm at a gathering and realize no one invited me",
        "i tell a story but no one is listening to me",
    ],
    "being_chased": [
        "something is chasing me but i'm running in slow motion",
        "footsteps follow me through dark hallways and get closer",
        "i'm being pursued but every door i try is locked",
        "a figure chases me and i can't see their face",
        "i run but my legs won't move fast enough",
        "something hunts me through empty streets at night",
        "i'm fleeing through a maze with no exit",
        "a shadow follows me no matter where i turn",
        "i'm running but the hallway keeps getting longer",
        "i'm being chased and i can hear breathing behind me",
    ],
    "falling": [
        "i'm falling from a tall building toward the ground",
        "the floor gives way beneath me and i'm dropping",
        "i slip from a ledge and can't grab anything",
        "i'm plummeting through darkness with no end",
        "i lose my balance on a high place and fall",
        "the ground disappears and i'm falling through space",
        "i stumble over an edge and begin to fall",
        "i'm on a tall structure that starts to collapse",
        "i'm falling and the ground rushes up toward me",
        "i trip and find myself falling down an endless shaft",
    ],
    "trapped": [
        "i'm in a small space and the walls are closing in",
        "i try to open the door but it won't budge",
        "i'm stuck in a room with no windows or exits",
        "the elevator stalls and i'm trapped inside",
        "i'm in a tight space and can't move forward or back",
        "i'm locked in somewhere and can't get out",
        "i crawl through a vent that gets narrower and narrower",
        "i'm in a room and the ceiling is lowering",
        "i'm trapped behind a door that's stuck shut",
        "i'm in a confined space and running out of air",
    ],
    "drowning": [
        "i'm underwater and can't find the surface",
        "water rises around me and i can't escape",
        "i'm sinking in deep water and can't swim up",
        "a wave covers my head and i can't breathe",
        "i'm in water over my head and going under",
        "the car is sinking and i'm trapped inside",
        "i'm in a pool but can't reach the edge",
        "water floods in and i'm running out of air",
        "i'm submerged and my lungs are burning",
        "i'm drowning and no one can see me",
    ],
    "vehicle": [
        "my car's brakes fail on a steep hill",
        "i'm driving but the steering wheel won't turn",
        "a truck swerves into my lane and i can't avoid it",
        "i'm in a vehicle heading toward a cliff edge",
        "the train isn't slowing down as it enters the station",
        "i'm driving but can't see where i'm going",
        "headlights rush toward me on a dark road",
        "i'm in a car that's accelerating on its own",
        "the vehicle is out of control and i can't stop it",
        "i'm driving and realize the brakes don't work",
    ],
    "loss_control": [
        "my body won't respond to my commands",
        "i try to speak but different words come out",
        "i'm trying to move but i'm paralyzed",
        "i can't make my hands do what i want",
        "i'm trying to run but moving in slow motion",
        "my voice won't work when i try to call for help",
        "i want to wake up but i'm stuck in the dream",
        "i reach for something but my arms won't obey",
        "i try to scream but no sound comes out",
        "i'm trying to stop but my body keeps moving",
    ],
    "abandonment": [
        "everyone disappears and i'm suddenly alone",
        "my friends leave the party without telling me",
        "i call out but no one responds or turns around",
        "i'm in a crowd but then everyone vanishes",
        "people i know walk past me like i'm invisible",
        "i arrive somewhere and the person i'm meeting never shows",
        "everyone leaves me behind without explanation",
        "i'm alone in a place that was just full of people",
        "i reach for support but everyone has turned away",
        "i'm left behind and can't find anyone i know",
    ],
}

# Threat keywords for each category
THREAT_KEYWORDS = {
    "exam_academic": ["test", "exam", "question", "failure"],
    "presentation": ["stage", "audience", "presentation", "performance"],
    "social_judgment": ["judgment", "stares", "mockery", "embarrassment"],
    "being_chased": ["pursuer", "footsteps", "chase", "hunter"],
    "falling": ["fall", "drop", "height", "ground"],
    "trapped": ["walls", "space", "confinement", "trap"],
    "drowning": ["water", "drowning", "surface", "breath"],
    "vehicle": ["vehicle", "brakes", "crash", "collision"],
    "loss_control": ["paralysis", "voice", "control", "movement"],
    "abandonment": ["isolation", "absence", "loneliness", "abandonment"],
}


def generate_comfort(nightmare_category, nightmare_text):
    """Generate IRT-based comfort response."""
    threat = random.choice(THREAT_KEYWORDS[nightmare_category])
    
    components = []
    
    # 70% chance of grounding
    if random.random() < 0.7:
        components.append(random.choice(GROUNDING))
    
    # Always mastery
    mastery = random.choice(MASTERY).format(threat=threat)
    components.append(mastery)
    
    # 80% chance of threat reduction
    if random.random() < 0.8:
        reduction = random.choice(THREAT_REDUCTION).format(threat=threat)
        components.append(reduction)
    
    # 50% chance of closure
    if random.random() < 0.5:
        components.append(random.choice(CLOSURE))
    
    return ' '.join(components)


def generate_dataset(target_size=1000):
    """Generate target_size unique nightmare-comfort pairs."""
    dataset = []
    seen_nightmares = set()
    
    # Generate from templates
    for category, templates in NIGHTMARE_TEMPLATES.items():
        for nightmare_base in templates:
            # Add variations
            variations = [
                nightmare_base,
                f"{nightmare_base} and panic sets in",
                f"{nightmare_base}; i freeze in fear",
                f"{nightmare_base}. my heart races",
                f"{nightmare_base}; i can't think clearly",
            ]
            
            for nightmare in variations:
                if len(dataset) >= target_size:
                    break
                
                if nightmare in seen_nightmares:
                    continue
                
                seen_nightmares.add(nightmare)
                
                # Generate 1-2 comfort variations per nightmare
                for _ in range(random.randint(1, 2)):
                    if len(dataset) >= target_size:
                        break
                    
                    comfort = generate_comfort(category, nightmare)
                    dataset.append({
                        "nightmare": nightmare,
                        "comfort_rewrite": comfort,
                        "category": category
                    })
    
    # If we need more, generate additional combinations
    while len(dataset) < target_size:
        category = random.choice(list(NIGHTMARE_TEMPLATES.keys()))
        nightmare = random.choice(NIGHTMARE_TEMPLATES[category])
        
        # Add random variation
        modifiers = [
            "and i can't escape",
            "; everything feels wrong",
            "but no one helps",
            "and time is running out",
            "; i'm all alone",
        ]
        nightmare = f"{nightmare} {random.choice(modifiers)}"
        
        if nightmare not in seen_nightmares:
            seen_nightmares.add(nightmare)
            comfort = generate_comfort(category, nightmare)
            dataset.append({
                "nightmare": nightmare,
                "comfort_rewrite": comfort,
                "category": category
            })
    
    return dataset[:target_size]


def main():
    print("="*70)
    print("GENERATING 1,000 IRT NIGHTMARE-COMFORT PAIRS")
    print("="*70)
    
    dataset = generate_dataset(1000)
    
    # Shuffle
    random.shuffle(dataset)
    
    # Save
    output_file = "data/irt_1000_pairs.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            # Remove category for training file
            train_item = {
                "nightmare": item["nightmare"],
                "comfort_rewrite": item["comfort_rewrite"]
            }
            f.write(json.dumps(train_item, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Generated {len(dataset)} unique pairs")
    print(f"✓ Saved to: {output_file}")
    
    # Statistics
    categories = {}
    for item in dataset:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nBreakdown by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Run: python merge_and_split.py")
    print("   (But change input file to irt_1000_pairs.jsonl)")
    print("2. Update config.yaml with new data paths")
    print("3. Train: python train_comfort_val.py")


if __name__ == "__main__":
    random.seed(42)
    main()