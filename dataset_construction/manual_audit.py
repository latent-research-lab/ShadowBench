"""
Given list of facts, the scripts prints each fact at a time in the CLI and allows users to either accept the fact or reject.

Fun way to audit the extracts facts and allows to only keep high-quality data for QA generation
"""


import json
import readline

def manual_review_tool(entity, sections="__all__"):
    try:
        with open(f"entity_facts/sports/tennis/upper/{entity.lower().replace(' ', '_')}.json", 'r') as f:
            data = json.load(f)
    except:
        print(f"{entity} does not exist")
        return
        
    final_db = set()

    print("--- MANUAL REVIEW MODE ---")
    print("Keys: [y] Keep, [n] Discard, [e] Edit, [q] Quit")

    print(f"\n\n=== ENTITY: {entity} ===")
    
    # Only more than 5 facts
    count = sum([len(v) for k, v in data.items()])
    if count < 5:
        print("Less than 5 facts, so skipping")
        return

    for section, facts in data.items():
        if isinstance(sections, list):
            if section.lower() not in [s.lower() for s in sections]:
                continue
        
        print(f"\n\n=== Section: {section} ===")
        
        kept_facts = set()
        
        for fact in facts:
            print(f"\nFact: {fact}")
            choice = input("Keep? > ").lower()
            
            if choice == 'q':
                break
            elif choice == 's':
                break
            elif choice == 'n':
                continue
            elif choice == 'y':
                kept_facts.add(fact)
            elif choice == 'e':
                readline.set_startup_hook(lambda: readline.insert_text(fact))
                new_fact = input("Edit: ")
                kept_facts.add(new_fact)
                readline.set_startup_hook(None)
                
        final_db = final_db.union(kept_facts)
        
        if choice == 'q':
            break

        if choice == 's':
            continue

    print(f"Facts collected: {len(final_db)}")

    save = ''
    while save.lower() not in ['y', 'n']:
        save = input("Save to file (Y/N): ")
    
    if save.lower() == 'y':
        default_filename = f"{entity.lower().replace(' ', '_')}_audited.json"
        readline.set_startup_hook(lambda: readline.insert_text(default_filename))
        filename = input("Filename: ")
        with open(f"entity_facts_audited/sports/tennis/old_dataset/upper/{filename}", 'w') as f:
            json.dump(list(final_db), f, indent=4)
        readline.set_startup_hook(None)

def main():
    entities = [] # List of entities to audit

    for entity in entities:

        audited_data = manual_review_tool(entity)

if __name__ == "__main__":
    main()
