# Git Task Agents

## Available Git Agents

### /auto-commit
**Purpose**: Automatically commit current changes with AI-generated descriptive message
**Usage**: `/auto-commit "Brief description of changes"`
**What it does**:
1. Analyzes current git diff
2. Generates descriptive commit message
3. Commits and pushes changes
4. Provides confirmation

### /smart-push  
**Purpose**: Intelligent commit and push with analysis
**Usage**: `/smart-push`
**What it does**:
1. Reviews all changed files
2. Creates categorized commit message (feat/fix/docs/etc.)
3. Commits with proper formatting
4. Pushes to current branch

### /quick-save
**Purpose**: Fast checkpoint commit for work in progress  
**Usage**: `/quick-save "checkpoint description"`
**What it does**:
1. Quick commit with WIP prefix
2. Includes timestamp
3. Pushes to backup/track progress

### /release-commit
**Purpose**: Formal release/milestone commit
**Usage**: `/release-commit "v1.2.0 - Added vLLM support"`
**What it does**:
1. Creates formal release commit
2. Tags the commit if version provided
3. Pushes with proper release formatting

## Examples

```bash
# After making changes to code
/auto-commit "Added error handling to model loading"

# Quick work-in-progress save
/quick-save "Working on remote vLLM client"

# Ready for deployment
/release-commit "v1.0.0 - Initial commerce POC release"
```