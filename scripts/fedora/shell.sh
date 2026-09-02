#!/usr/bin/env bash
set -e

echo "==> Installing terminal packages..."

sudo dnf install -y \
    zsh \
    fzf \
    bat \
    eza \
    zoxide \
    git \
    curl \
    util-linux-user

echo "==> Installing Starship..."

if ! command -v starship >/dev/null 2>&1; then
    curl -sS https://starship.rs/install.sh | sh -s -- -y
fi

echo "==> Installing zsh plugins..."

ZSH_CUSTOM="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"

# Install Oh My Zsh if it doesn't exist
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    RUNZSH=no CHSH=no KEEP_ZSHRC=yes \
        sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
fi

mkdir -p "$ZSH_CUSTOM/plugins"

# Autosuggestions
if [ ! -d "$ZSH_CUSTOM/plugins/zsh-autosuggestions" ]; then
    git clone \
        https://github.com/zsh-users/zsh-autosuggestions \
        "$ZSH_CUSTOM/plugins/zsh-autosuggestions"
fi

# Syntax highlighting
if [ ! -d "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting" ]; then
    git clone \
        https://github.com/zsh-users/zsh-syntax-highlighting \
        "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting"
fi

echo "==> Creating .zshrc..."

cat > "$HOME/.zshrc" <<'EOF'
# ============================================================
# ZSH
# ============================================================

export EDITOR="code --wait"
export VISUAL="$EDITOR"

# Oh My Zsh
export ZSH="$HOME/.oh-my-zsh"

ZSH_THEME=""

plugins=(
    git
    fzf
    zsh-autosuggestions
    zsh-syntax-highlighting
)

source "$ZSH/oh-my-zsh.sh"


# ============================================================
# STARSHIP
# ============================================================

eval "$(starship init zsh)"


# ============================================================
# ZOXIDE
# ============================================================

eval "$(zoxide init zsh)"


# ============================================================
# FZF
# ============================================================

source <(fzf --zsh)


# ============================================================
# ALIASES
# ============================================================

# Modern ls
alias ls='eza --icons --group-directories-first'
alias ll='eza -lah --icons --group-directories-first'
alias la='eza -a --icons --group-directories-first'
alias lt='eza --tree --level=2 --icons'

# Better cat
alias cat='bat'

# Git
alias lg='lazygit'
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline --graph --decorate'

# Navigation
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'

# Common
alias c='clear'
alias v='code .'


# ============================================================
# PYTHON
# ============================================================

# Don't create .pyc files
export PYTHONDONTWRITEBYTECODE=1


# ============================================================
# HISTORY
# ============================================================

HISTSIZE=10000
SAVEHIST=10000

setopt HIST_IGNORE_DUPS
setopt HIST_IGNORE_SPACE
setopt SHARE_HISTORY


# ============================================================
# COMPLETION
# ============================================================

autoload -Uz compinit
compinit


# ============================================================
# KEY BINDINGS
# ============================================================

# Use Ctrl+R for fzf history search
bindkey '^R' fzf-history-widget

# Better word navigation
bindkey '^[[1;5C' forward-word
bindkey '^[[1;5D' backward-word
EOF


# ============================================================
# STARSHIP CONFIG
# ============================================================

mkdir -p "$HOME/.config"

cat > "$HOME/.config/starship.toml" <<'EOF'
add_newline = true

format = """
$directory\
$git_branch\
$git_status\
$python\
$cmd_duration
$character"""

[directory]
style = "bold cyan"
truncation_length = 3
truncate_to_repo = true

[git_branch]
format = " [$symbol$branch]($style)"
symbol = "󰘬 "
style = "bold purple"

[git_status]
format = " [$all_status$ahead_behind]($style)"
style = "bold red"

[python]
format = " [ $virtualenv]($style)"
style = "bold yellow"

[cmd_duration]
min_time = 2000
format = " [$duration]($style)"
style = "dimmed white"

[character]
success_symbol = "[❯](bold green)"
error_symbol = "[❯](bold red)"
EOF


# ============================================================
# DEFAULT SHELL
# ============================================================



ZSH_PATH="$(which zsh)"

if [ "$SHELL" != "$ZSH_PATH" ]; then
    echo "==> Changing default shell to zsh..."
    chsh -s "$ZSH_PATH"
fi
❯ # Create a dedicated directory for Nerd Fonts
mkdir -p ~/.local/share/fonts/NerdFonts

# Download and extract JetBrains Mono Nerd Font
curl -fLo "$HOME/.local/share/fonts/NerdFonts/JetBrainsMono.zip" \
    https://github.com/ryanoasis/nerd-fonts/releases/latest/download/JetBrainsMono.zip

unzip -o "$HOME/.local/share/fonts/NerdFonts/JetBrainsMono.zip" -d "$HOME/.local/share/fonts/NerdFonts/"
rm "$HOME/.local/share/fonts/NerdFonts/JetBrainsMono.zip"

# Rebuild font cache
fc-cache -f -v



echo
echo "============================================================"
echo " Terminal setup complete!"
echo "============================================================"
echo
echo "Restart VS Code or open a new terminal."
echo
echo "Useful commands:"
echo "  ll       -> detailed directory listing"
echo "  lt       -> directory tree"
echo "  z foo    -> jump to frequently used directory"
echo "  lg       -> lazygit"
echo "  Ctrl+R   -> fuzzy command history"
echo


# Users Settings (JSON)



#{
#    "terminal.integrated.defaultProfile.linux": "zsh",
#
#    "terminal.integrated.profiles.linux": {
#        "zsh": {
#            "path": "/usr/bin/zsh",
#            "icon": "terminal"
#        }
#    },
#
#    "terminal.integrated.fontFamily": "JetBrainsMono Nerd Font",
#    "terminal.integrated.fontSize": 14,
#    "terminal.integrated.lineHeight": 1.2,
#
#    "terminal.integrated.cursorStyle": "line",
#    "terminal.integrated.cursorBlinking": true,
#
#    "terminal.integrated.scrollback": 10000,
#
#    "terminal.integrated.enablePersistentSessions": true,
#    "terminal.integrated.persistentSessionScrollback": 1000,
#
#    "terminal.integrated.tabs.enabled": true,
#    "terminal.integrated.tabs.location": "right",
#
#    "terminal.integrated.confirmOnExit": "never",
#
#    "terminal.integrated.copyOnSelection": false,
#    "terminal.integrated.rightClickBehavior": "copyPaste",
#
#    "terminal.integrated.shellIntegration.enabled": true,
#
#    "terminal.integrated.cwd": "${workspaceFolder}",
#
#    "terminal.integrated.commandsToSkipShell": [
#        "workbench.action.quickOpenView"
#    ]
#}


