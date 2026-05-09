"""
LLM Prompt Templates for Retrosynthesis Planning
"""

def get_initial_prompt():
    """Get the initial system prompt for retrosynthesis"""
    return '''
        You are a professional chemist specializing in synthesis analysis. Your task is to propose a retrosynthesis route for a target molecule provided in SMILES format.

        Definition:
        A retrosynthesis route is a sequence of backward reactions that starts from the target molecules and ends with commercially purchasable building blocks.

        Key concepts:
        - Molecule set: The working set of molecules at any given step. Initially, it contains only the target molecule.
        - Commercially purchasable: Molecules that can be directly bought from suppliers (permitted building blocks).
        - Non-purchasable: Molecules that must be further decomposed via retrosynthesis steps.
        - Reaction source: All reactions must be derived from the USPTO dataset, and stereochemistry (e.g., E/Z isomers, chiral centers) must be preserved.

        Process:
        1. Initialization: Start with the molecule set = [target molecule].
        2. Iteration:
            - Select one non-purchasable molecule from the molecule set (the product).
            - Apply a valid backward reaction from the USPTO dataset to decompose it into reactants.
            - Remove the product molecule from the set.
            - Add the reactants to the set.
        3. Termination: Continue until all molecules in the set are commercially purchasable.
        '''


def get_task_description(target_smiles: str, examples: str, expansion_routes: int):
    """Format the task description with target and examples"""
    return f'''
        My target molecule is: {target_smiles}

        To assist you with the format, example retrosynthesis routes are provided:
        {examples}

        Please propose {expansion_routes} different retrosynthesis routes for my target molecule. The provided reference routes may be helpful.
        '''


def get_requirements():
    """Get the formatting requirements for the response"""
    return '''
        You need to analyze the target molecule and make a retrosynthesis plan in the <PLAN></PLAN> before proposing the route. After making the plan, you should explain the plan in the <EXPLANATION></EXPLANATION>. The route should be a list of steps wrapped in <ROUTE></ROUTE>. Each step in the list should be a dictionary.
        At the first step, the molecule set should be the target molecules set given by the user. Here is an example:

        <PLAN>: Analyze the target molecule and plan for each step in the route. </PLAN>
        <EXPLANATION>: Explain the plan. </EXPLANATION>
        <ROUTE>
        [
            {
                'Molecule set': "['CCOc1cc2ncnc(Nc3ccc4c(cnn4Cc4ccccc4)c3)c2cc1OCCOC']",
                'Rational': "Disconnect the ether linkage at the ethoxy group to simplify the aromatic structure",
                'Product': "['CCOc1cc2ncnc(Nc3ccc4c(cnn4Cc4ccccc4)c3)c2cc1OCCOC']",
                'Reaction': "['CCOc1cc2ncnc(Nc3ccc4c(cnn4Cc4ccccc4)c3)c2cc1OCCOC>>CCOc1cc2ncnc(Nc3ccc4c(cnn4Cc4ccccc4)c3)c2cc1O.CCOC']",
                'Reactants': "['CCOc1cc2ncnc(Nc3ccc4c(cnn4Cc4ccccc4)c3)c2cc1O', 'CCOC']",
                'Updated molecule set': "['CCOc1cc2ncnc(Nc3ccc4c(cnn4Cc4ccccc4)c3)c2cc1O', 'CCOC']"
            },
            {
                'Molecule set': "['CCOc1cc2ncnc(Nc3ccc4c(cnn4Cc4ccccc4)c3)c2cc1O', 'CCOC']",
                'Rational': "Deprotect the phenol to prepare for further functionalization",
                'Product': "['CCOc1cc2ncnc(Nc3ccc4c(cnn4Cc4ccccc4)c3)c2cc1O']",
                'Reaction': "['CCOc1cc2ncnc(Nc3ccc4c(cnn4Cc4ccccc4)c3)c2cc1O>>Cc1cc2ncnc(Nc3ccc4c(cnn4Cc4ccccc4)c3)c2cc1O.CCO']",
                'Reactants': "['Cc1cc2ncnc(Nc3ccc4c(cnn4Cc4ccccc4)c3)c2cc1O', 'CCO']",
                'Updated molecule set': "['Cc1cc2ncnc(Nc3ccc4c(cnn4Cc4ccccc4)c3)c2cc1O', 'CCO', 'CCOC']"
            }
        ]
        </ROUTE>

        \n\n
        CRITICAL FORMATTING RULES - THE OUTPUT MUST BE VALID PYTHON CODE:
        1. ALL dictionary values MUST be strings wrapped in double quotes: "..."
        2. Lists must be INSIDE strings: "['item1', 'item2']" NOT ['item1', 'item2']
        3. SMILES strings must have quotes inside the string: "['CCO']" NOT "[CCO]"
        4. The 'Rational' field MUST be a quoted string: "This is my reasoning" NOT This is my reasoning
        5. The entire route must be valid Python code that can be parsed by ast.literal_eval()

        VALID FORMAT EXAMPLE:
        {
            'Rational': "Disconnect the ether linkage",  # ← QUOTED STRING
            'Product': "['CCOc1ccccc1']",                 # ← String containing list
            'Reactants': "['CCO', 'Oc1ccccc1']",         # ← String containing list
        }

        INVALID FORMAT (DO NOT USE):
        {
            'Rational': Disconnect the ether linkage,    # ← NO QUOTES = SYNTAX ERROR
            'Product': ['CCOc1ccccc1'],                   # ← Direct list = PARSING ERROR
            'Reactants': [CCO, Oc1ccccc1],               # ← Missing quotes = SYNTAX ERROR
        }

        CRITICAL SMILES FORMATTING - USE CANONICAL SMILES:
        ⚠️  ALWAYS use canonical SMILES notation for all molecules to ensure consistency with the reaction template engine.

        What is canonical SMILES?
        - Canonical SMILES is a standardized representation where a molecule has exactly ONE correct SMILES string
        - RDKit automatically generates canonical SMILES (the format used by USPTO reaction templates)
        - Using canonical SMILES prevents validation failures due to notation differences

        Examples of canonical vs non-canonical SMILES:
        ✅ CORRECT (Canonical):   "CCO"              (ethanol)
        ❌ WRONG (Non-canonical): "C(C)O"            (same molecule, wrong notation)
        ❌ WRONG (Non-canonical): "OCC"              (same molecule, wrong notation)

        ✅ CORRECT (Canonical):   "CC(C)O"           (isopropanol)
        ❌ WRONG (Non-canonical): "C(C)(C)O"         (same molecule, wrong notation)

        ✅ CORRECT (Canonical):   "c1ccccc1"         (benzene - aromatic)
        ❌ WRONG (Non-canonical): "C1=CC=CC=C1"      (same molecule, wrong notation)

        ✅ CORRECT (Canonical):   "C1CCCCC1"         (cyclohexane - aliphatic)
        ❌ WRONG (Non-canonical): "C2CCCC2C"         (same molecule, wrong notation)

        HOW TO ENSURE CANONICAL SMILES:
        1. Use the most compact representation (fewer characters when possible)
        2. For aromatics, use lowercase letters: c, n, o, s (e.g., "c1ccccc1" not "C1=CC=CC=C1")
        3. Start numbering from 1, use minimal ring closure numbers
        4. Follow standard atom ordering (heavy atoms before hydrogen)
        5. When in doubt, mentally convert to RDKit canonical form

        WHY THIS MATTERS:
        - The retrosynthesis engine uses canonical SMILES from USPTO templates
        - Non-canonical SMILES will cause validation failures even if the chemistry is correct
        - Using canonical SMILES from the start ensures smooth route validation

        Requirements: 1. The 'Molecule set' contains molecules we need to synthesize at this stage. In the first step, it should be the target molecule. In the following steps, it should be the 'Updated molecule set' from the previous step. It must be a STRING containing a list: "['SMILES1', 'SMILES2']"\n
        2. The 'Rational' part in each step should be your analysis for synthesis planning in this step. It MUST be a quoted string: "Your analysis here"\n
        3. 'Product' is the molecule we plan to synthesize in this step. It should be from the 'Molecule set'. It must be a STRING containing a list with one molecule: "['SMILES']"\n
        4. 'Reaction' is a backward reaction which can decompose the product molecule into its reactants. It must be a STRING containing a list: "['Product>>Reactant1.Reactant2']"\n
        5. 'Reactants' are the reactants of the reaction. It must be a STRING containing a list: "['SMILES1', 'SMILES2']"\n
        6. The 'Updated molecule set' should be molecules we need to purchase or synthesize after taking this reaction. To get the 'Updated molecule set', you need to remove the product molecule from the 'Molecule set' and then add the reactants in this step into it. In the last step, all the molecules in the 'Updated molecule set' should be purchasable. It must be a STRING containing a list: "['SMILES1', 'SMILES2']"\n
        7. In the <PLAN>, you should analyze the target molecule and plan for the whole route.\n
        8. In the <EXPLANATION>, you should analyze the plan.\n
        9. REMEMBER: Every value in the dictionary must be a string wrapped in quotes. The route must be parseable by Python's ast.literal_eval() function.\n
        10. CRITICAL: Use ONLY canonical SMILES notation for all molecules (Product, Reactants, Molecule set, Updated molecule set, and in Reactions).\n'''


def construct_full_prompt(target_smiles: str, examples: str, expansion_routes: int):
    """Combine all prompt parts into complete prompt"""
    initial_prompt = get_initial_prompt()
    task_description = get_task_description(target_smiles, examples, expansion_routes)
    requirements = get_requirements()
    
    return initial_prompt + task_description + requirements