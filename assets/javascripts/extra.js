(function() {
    // Init: Add "Copy to LLM" button
    function init() {
        createCopyToLLMButton();
    }

    // Add "Copy to LLM" button to page title
    function createCopyToLLMButton() {
        if (document.querySelector('.copy-llm-inline-container')) return;

        const h1 = document.querySelector('.md-content__inner h1');
        if (!h1) return;

        const btnContainer = document.createElement('span');
        btnContainer.className = 'copy-llm-inline-container';
        
        const btn = document.createElement('button');
        btn.className = 'md-icon copy-llm-btn-inline';
        btn.setAttribute('aria-label', 'Copy page for LLM');
        btn.setAttribute('title', 'Copy content as Markdown for LLM');
        
        btn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16">
                <path d="M19 3h-4.18C14.4 1.84 13.3 1 12 1s-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2m-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1m2 14H7v-2h7zm3-4H7v-2h10zm0-4H7V7h10z"/>
            </svg>
            <span class="copy-llm-text">Copy page</span>
        `;

        // Convert HTML content to Markdown
        function htmlToMarkdown(element) {
            if (!element) return '';
            
            const clone = element.cloneNode(true);
            
            // Cleanup unwanted elements
            ['.md-clipboard', '.md-nav', '.headerlink', '.md-source-file', '.md-footer', '.copy-llm-inline-container', 'script', 'style', 'noscript', 'iframe', 'svg']
                .forEach(sel => clone.querySelectorAll(sel).forEach(el => el.remove()));

            function convert(node) {
                if (node.nodeType === Node.TEXT_NODE) return node.textContent;
                if (node.nodeType !== Node.ELEMENT_NODE) return '';

                const tag = node.tagName.toLowerCase();
                const classes = Array.from(node.classList);

                // Handle Admonitions
                if (classes.some(c => c.startsWith('admonition') || c === 'admonition')) {
                    const titleEl = node.querySelector('.admonition-title');
                    const title = titleEl ? titleEl.textContent.trim() : 'Note';
                    
                    let content = '';
                    node.childNodes.forEach(child => {
                        if (child.nodeType === Node.ELEMENT_NODE && child.classList.contains('admonition-title')) return;
                        content += convert(child);
                    });
                    
                    return `> **${title}**\n>\n> ${content.trim().replace(/\n/g, '\n> ')}\n\n`;
                }

                let content = '';
                node.childNodes.forEach(child => content += convert(child));

                switch (tag) {
                    case 'h1': return `# ${content.trim()}\n\n`;
                    case 'h2': return `## ${content.trim()}\n\n`;
                    case 'h3': return `### ${content.trim()}\n\n`;
                    case 'h4': return `#### ${content.trim()}\n\n`;
                    case 'h5': return `##### ${content.trim()}\n\n`;
                    case 'h6': return `###### ${content.trim()}\n\n`;
                    case 'p': return `${content.trim()}\n\n`;
                    case 'br': return `\n`;
                    case 'hr': return `\n---\n\n`;
                    case 'strong': case 'b': return `**${content}**`;
                    case 'em': case 'i': return `*${content}*`;
                    case 'code':
                        return node.parentElement.tagName.toLowerCase() === 'pre' ? content : `\`${content}\``;
                    case 'pre':
                        const code = node.querySelector('code');
                        const lang = code?.className.split(' ').find(c => c.startsWith('language-'))?.replace('language-', '') || '';
                        return `\n\`\`\`${lang}\n${(code || node).textContent.trim()}\n\`\`\`\n\n`;
                    case 'ul': case 'ol': case 'dl': return `${content}\n`;
                    case 'li':
                        const prefix = node.parentElement?.tagName.toLowerCase() === 'ol' 
                            ? `${Array.from(node.parentElement.children).indexOf(node) + 1}. ` 
                            : '- ';
                        return `${prefix}${content.trim()}\n`;
                    case 'blockquote': return `> ${content.trim().replace(/\n/g, '\n> ')}\n\n`;
                    case 'dt': return `**${content.trim()}**\n`;
                    case 'dd': return `: ${content.trim()}\n\n`;
                    case 'a': return content.trim() ? `[${content}](${node.getAttribute('href')})` : content;
                    case 'img': 
                        const title = node.getAttribute('title');
                        return `![${node.getAttribute('alt') || ''}](${node.getAttribute('src')}${title ? ` "${title}"` : ''})`;
                    case 'table':
                        const rows = Array.from(node.querySelectorAll('tr'));
                        if (!rows.length) return '';
                        const headers = Array.from(rows[0].querySelectorAll('th, td'));
                        const body = rows.slice(1);
                        return `\n| ${headers.map(h => h.textContent.trim()).join(' | ')} |\n| ${headers.map(() => '---').join(' | ')} |\n${body.map(r => '| ' + Array.from(r.querySelectorAll('td, th')).map(c => c.textContent.trim()).join(' | ') + ' |').join('\n')}\n\n`;
                    default: return content;
                }
            }
            
            return convert(clone).replace(/\n{3,}/g, '\n\n').trim();
        }

        btn.addEventListener('click', async () => {
            try {
                const contentInner = document.querySelector('.md-content__inner');
                if (!contentInner) return;

                const text = htmlToMarkdown(contentInner);
                
                const llmContent = `# Context from langchain-dev-utils docs \n\n${text}`;

                await navigator.clipboard.writeText(llmContent);

                const originalHTML = btn.innerHTML;
                btn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16">
                        <path d="M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                    </svg>
                    <span class="copy-llm-text">Copied!</span>
                `;
                btn.classList.add('copy-success');
                
                setTimeout(() => {
                    btn.innerHTML = originalHTML;
                    btn.classList.remove('copy-success');
                }, 2000);

            } catch (err) {
                console.error('Failed to copy: ', err);
                alert('Failed to copy content to clipboard');
            }
        });

        btnContainer.appendChild(btn);
        h1.prepend(btnContainer);
    }

    // Initialize
    document.addEventListener("DOMContentLoaded", init);

    // Re-init on navigation (MkDocs instant loading support)
    if (window.document$) {
        window.document$.subscribe(function() {
            setTimeout(init, 100); 
        });
    }
})();
