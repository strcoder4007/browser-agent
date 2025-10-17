import pdb

import pyperclip
from typing import Optional, Type, Callable, Dict, Any, Union, Awaitable, TypeVar
from pydantic import BaseModel
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller, DoneAction
from browser_use.controller.registry.service import Registry, RegisteredAction
from main_content_extractor import MainContentExtractor
from browser_use.controller.views import (
    ClickElementAction,
    DoneAction,
    ExtractPageContentAction,
    GoToUrlAction,
    InputTextAction,
    OpenTabAction,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)
import logging
import inspect
import asyncio
import os
from langchain_core.language_models.chat_models import BaseChatModel
from browser_use.agent.views import ActionModel, ActionResult

from src.utils.mcp_client import create_tool_param_model, setup_mcp_client_and_tools

from browser_use.utils import time_execution_sync

logger = logging.getLogger(__name__)

Context = TypeVar('Context')


class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [],
                 output_model: Optional[Type[BaseModel]] = None,
                 ask_assistant_callback: Optional[Union[Callable[[str, BrowserContext], Dict[str, Any]], Callable[
                     [str, BrowserContext], Awaitable[Dict[str, Any]]]]] = None,
                 ):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()
        self.ask_assistant_callback = ask_assistant_callback
        self.mcp_client = None
        self.mcp_server_config = None

    def _register_custom_actions(self):
        """Register all custom browser actions"""

        @self.registry.action(
            "When executing tasks, prioritize autonomous completion. However, if you encounter a definitive blocker "
            "that prevents you from proceeding independently – such as needing credentials you don't possess, "
            "requiring subjective human judgment, needing a physical action performed, encountering complex CAPTCHAs, "
            "or facing limitations in your capabilities – you must request human assistance."
        )
        async def ask_for_assistant(query: str, browser: BrowserContext):
            if self.ask_assistant_callback:
                if inspect.iscoroutinefunction(self.ask_assistant_callback):
                    user_response = await self.ask_assistant_callback(query, browser)
                else:
                    user_response = self.ask_assistant_callback(query, browser)
                msg = f"AI ask: {query}. User response: {user_response['response']}"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                return ActionResult(extracted_content="Human cannot help you. Please try another way.", include_in_memory=True)

        @self.registry.action('Upload file to interactive element with file path ')
        async def upload_file(index: int, path: str, browser: BrowserContext, available_file_paths: list[str]):
            if path not in available_file_paths:
                return ActionResult(error=f'File path {path} is not available')
            if not os.path.exists(path):
                return ActionResult(error=f'File {path} does not exist')
            dom_el = await browser.get_dom_element_by_index(index)
            file_upload_dom_el = dom_el.get_file_upload_element()
            if file_upload_dom_el is None:
                msg = f'No file upload element found at index {index}'
                logger.info(msg)
                return ActionResult(error=msg)
            file_upload_el = await browser.get_locate_element(file_upload_dom_el)
            if file_upload_el is None:
                msg = f'No file upload element found at index {index}'
                logger.info(msg)
                return ActionResult(error=msg)
            try:
                await file_upload_el.set_input_files(path)
                msg = f'Successfully uploaded file to index {index}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                msg = f'Failed to upload file to index {index}: {str(e)}'
                logger.info(msg)
                return ActionResult(error=msg)

        @self.registry.action("Read clipboard content and return it as extracted content. Use this after clicking 'Copy link to post'.")
        async def read_clipboard(browser: BrowserContext):
            try:
                text = pyperclip.paste()
                if not text:
                    msg = 'Clipboard is empty.'
                    logger.info(msg)
                    return ActionResult(error=msg)
                logger.info(f'Read clipboard text: {text[:200]}')
                return ActionResult(extracted_content=text, include_in_memory=True)
            except Exception as e:
                msg = f'Failed to read clipboard: {str(e)}'
                logger.info(msg)
                return ActionResult(error=msg)

        @self.registry.action(
            "Aggregate LinkedIn posts into a single JSON file and/or return merged JSON. "
            "Accepts JSON from linkedin_extract_recent_activity, merges with existing file, "
            "deduplicates by link, and saves to ./links_raw.json by default."
        )
        async def aggregate_posts(posts_json: str, save_path: str = "./links_raw.json", dedupe_by_link: bool = True, return_merged: bool = True):
            import os, json
            try:
                incoming = json.loads(posts_json)
            except Exception:
                try:
                    import json_repair
                    incoming = json.loads(json_repair.repair_json(posts_json))
                except Exception as e:
                    return ActionResult(error=f"Invalid posts_json for aggregation: {e}")
            if isinstance(incoming, dict) and isinstance(incoming.get('posts'), list):
                new_posts = incoming['posts']
            elif isinstance(incoming, list):
                new_posts = incoming
            else:
                return ActionResult(error='posts_json must be an array or an object with key "posts"')
            aggregated = []
            try:
                if os.path.exists(save_path):
                    with open(save_path, 'r', encoding='utf-8') as fr:
                        data = json.load(fr)
                        if isinstance(data, dict) and isinstance(data.get('posts'), list):
                            aggregated = data['posts']
                        elif isinstance(data, list):
                            aggregated = data
            except Exception:
                pass
            seen = set()
            for p in aggregated:
                link = (p or {}).get('link')
                if link:
                    seen.add(link)
            for p in new_posts:
                link = (p or {}).get('link')
                if dedupe_by_link and link and link in seen:
                    continue
                aggregated.append(p)
                if link:
                    seen.add(link)
            try:
                dirn = os.path.dirname(save_path)
                if dirn:
                    os.makedirs(dirn, exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as fw:
                    json.dump({'posts': aggregated}, fw, ensure_ascii=False, indent=2)
            except Exception as e:
                return ActionResult(error=f"Failed to save aggregated posts to {save_path}: {e}")
            content = {'posts': aggregated} if return_merged else {'saved_path': save_path}
            return ActionResult(extracted_content=json.dumps(content, ensure_ascii=False), include_in_memory=True)

        @self.registry.action(
            "Extract LinkedIn recent-activity posts using Playwright. Do NOT try to figure out UI clicks yourself. "
            "This action finds each post's three-dots menu, clicks 'Copy link to post', reads the clipboard for the link, "
            "and also extracts the post text and its relative timestamp. Returns JSON with a list of {link, text, age}. "
            "Optional parameters: max_posts (default 10)."
        )
        async def linkedin_extract_recent_activity(browser: BrowserContext, max_posts: int = 10):
            try:
                try:
                    from playwright.async_api import Page, BrowserContext as PWBrowserContext  # type: ignore
                except Exception:
                    Page = None
                    PWBrowserContext = None
                page = None
                candidate_attrs = ['page', '_page', 'current_page', 'playwright_page', 'context', '_context', 'playwright_context', 'pw_context']
                for attr in candidate_attrs:
                    obj = getattr(browser, attr, None)
                    if obj is None:
                        continue
                    if Page is not None:
                        try:
                            from playwright.async_api import Page as _P
                            if isinstance(obj, _P):
                                page = obj
                                break
                        except Exception:
                            pass
                    if PWBrowserContext is not None:
                        try:
                            from playwright.async_api import BrowserContext as _C
                            if isinstance(obj, _C):
                                pages = obj.pages
                                if pages:
                                    page = pages[-1]
                                    break
                        except Exception:
                            pass
                    if hasattr(obj, 'pages'):
                        try:
                            pages = obj.pages
                            if pages:
                                page = pages[-1]
                                break
                        except Exception:
                            pass
                if page is None and hasattr(browser, 'get_current_page'):
                    try:
                        maybe_page = await browser.get_current_page()  # type: ignore
                        page = maybe_page
                    except Exception:
                        pass
                if page is None:
                    msg = 'Unable to access Playwright page from BrowserContext.'
                    logger.error(msg)
                    return ActionResult(error=msg)
                posts_data = []
                three_dots_selector = "button.feed-shared-control-menu__trigger"
                try:
                    await page.wait_for_selector(three_dots_selector, timeout=5000)
                except Exception:
                    pass
                buttons = page.locator(three_dots_selector)
                try:
                    count = await buttons.count()
                except Exception:
                    count = 0
                if count == 0:
                    msg = 'No three-dots menu buttons found on this page.'
                    logger.info(msg)
                    return ActionResult(error=msg)
                # Hard cap: never scrape more than 5 posts per batch regardless of parameter
                HARD_POST_LIMIT = 5
                try:
                    user_limit = int(max_posts)
                except Exception:
                    user_limit = HARD_POST_LIMIT
                total = min(count, HARD_POST_LIMIT, max(1, user_limit))
                for i in range(total):
                    try:
                        btn = buttons.nth(i)
                        try:
                            await btn.scroll_into_view_if_needed()
                            await asyncio.sleep(0.1)
                        except Exception:
                            pass
                        content_text = None
                        age_text = None
                        # Find post root
                        try:
                            post_root = btn.locator(
                                "xpath=ancestor::*[self::article or contains(@class,'feed-shared-update') or contains(@class,'update') or @data-urn][1]"
                            )
                            if await post_root.count() == 0:
                                post_root = btn
                        except Exception:
                            post_root = btn
                        # Expand 'See more' if present
                        try:
                            see_more = post_root.locator("button:has-text('See more')")
                            if await see_more.count() > 0:
                                try:
                                    await see_more.first.click(timeout=1000)
                                    await asyncio.sleep(0.05)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # Extract content via robust selectors
                        try:
                            content_loc = post_root.locator(
                                "css=div.update-components-text.relative.update-components-update-v2__commentary"
                            )
                            if await content_loc.count() == 0:
                                content_loc = post_root.locator(
                                    "css=div.update-components-text.update-components-update-v2__commentary"
                                )
                            if await content_loc.count() == 0:
                                content_loc = post_root.locator(
                                    "xpath=.//div[contains(@class,'update-components-text') and contains(@class,'update-components-update-v2__commentary')]"
                                )
                            if await content_loc.count() > 0:
                                # Try inner_text of container
                                try:
                                    txt = (await content_loc.first.inner_text()).strip()
                                except Exception:
                                    txt = None
                                if not txt:
                                    # Concatenate spans
                                    try:
                                        spans = content_loc.first.locator('span')
                                        texts = await spans.all_inner_texts()
                                        txt = ' '.join([t.strip() for t in texts if t and t.strip()]) or None
                                    except Exception:
                                        txt = None
                                if not txt:
                                    try:
                                        txt = await content_loc.first.evaluate("el => el.innerText || el.textContent")
                                        if txt:
                                            txt = txt.strip()
                                    except Exception:
                                        txt = None
                                content_text = txt
                        except Exception:
                            content_text = None
                        # Extract timestamp under the specified container
                        try:
                            ts_container = post_root.locator(
                                "css=div.update-components-actor__sub-description.text-body-xsmall"
                            )
                            ts_loc = ts_container.locator('span') if await ts_container.count() > 0 else post_root.locator(
                                "xpath=.//div[contains(@class,'update-components-actor__sub-description') and contains(@class,'text-body-xsmall')]//span"
                            )
                            if await ts_loc.count() > 0:
                                try:
                                    age_text = (await ts_loc.first.inner_text()).strip()
                                except Exception:
                                    try:
                                        age_text = await ts_loc.first.evaluate("el => el.innerText || el.textContent")
                                        if age_text:
                                            age_text = age_text.strip()
                                    except Exception:
                                        age_text = None
                            if not age_text:
                                # Fallback to <time>
                                try:
                                    time_el = post_root.locator('time')
                                    if await time_el.count() > 0:
                                        age_text = (await time_el.first.inner_text()).strip()
                                except Exception:
                                    pass
                        except Exception:
                            age_text = None
                        try:
                            await btn.click(timeout=3000)
                        except Exception:
                            try:
                                await btn.click(timeout=3000, force=True)
                            except Exception as e:
                                logger.info(f"Failed to click three-dots for item {i}: {e}")
                                continue
                        copied = False
                        try:
                            await page.wait_for_selector("text=Copy link to post", timeout=3000)
                            menu_item = page.locator("text=Copy link to post").first
                            await menu_item.click()
                            copied = True
                        except Exception:
                            try:
                                menu_item = page.locator("[role='menu'] :text('Copy link to post')").first
                                if await menu_item.count() > 0:
                                    await menu_item.click()
                                    copied = True
                            except Exception:
                                copied = False
                        link = None
                        if copied:
                            await asyncio.sleep(0.2)
                            try:
                                link = pyperclip.paste() or None
                            except Exception:
                                link = None
                        posts_data.append({'link': link, 'text': content_text, 'age': age_text})
                        try:
                            await page.keyboard.press('Escape')
                        except Exception:
                            pass
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        logger.info(f"Error processing post index {i}: {e}")
                        posts_data.append({'link': None, 'text': None, 'age': None, 'error': str(e)})
                try:
                    import json as _json
                    result_text = _json.dumps({'posts': posts_data}, ensure_ascii=False)
                except Exception:
                    result_text = str({'posts': posts_data})
                return ActionResult(extracted_content=result_text, include_in_memory=True)
            except Exception as e:
                msg = f'Failed to extract LinkedIn recent-activity posts: {str(e)}'
                logger.error(msg)
                return ActionResult(error=msg)

        @self.registry.action(
            "Extract recent-activity posts and save to file in one step. Uses Playwright to click/copy links, "
            "extracts text and age, then merges into an aggregate JSON file with de-duplication by link."
        )
        async def linkedin_extract_and_save(
            browser: BrowserContext,
            max_posts: int = 10,
            aggregate_path: str = "./links_raw.json",
            dedupe_by_link: bool = True,
        ):
            import os, json
            # Use the existing extractor to get a batch
            batch_res = await linkedin_extract_recent_activity(browser=browser, max_posts=max_posts)
            if batch_res.error:
                return batch_res
            try:
                batch_obj = json.loads(batch_res.extracted_content) if batch_res.extracted_content else {"posts": []}
            except Exception:
                try:
                    import json_repair
                    batch_obj = json.loads(json_repair.repair_json(batch_res.extracted_content or "{}"))
                except Exception as e:
                    return ActionResult(error=f"Failed to parse extraction batch: {e}")

            new_posts = []
            if isinstance(batch_obj, dict) and isinstance(batch_obj.get('posts'), list):
                new_posts = batch_obj['posts']
            elif isinstance(batch_obj, list):
                new_posts = batch_obj

            # Load existing aggregate
            aggregated = []
            try:
                if os.path.exists(aggregate_path):
                    with open(aggregate_path, 'r', encoding='utf-8') as fr:
                        data = json.load(fr)
                        if isinstance(data, dict) and isinstance(data.get('posts'), list):
                            aggregated = data['posts']
                        elif isinstance(data, list):
                            aggregated = data
            except Exception:
                pass

            before_count = len(aggregated)
            seen = { (p or {}).get('link') for p in aggregated if (p or {}).get('link') } if dedupe_by_link else set()

            added = 0
            for p in new_posts:
                link = (p or {}).get('link')
                if dedupe_by_link and link and link in seen:
                    continue
                aggregated.append(p)
                if link:
                    seen.add(link)
                added += 1

            # Save aggregate
            try:
                dirn = os.path.dirname(aggregate_path)
                if dirn:
                    os.makedirs(dirn, exist_ok=True)
                with open(aggregate_path, 'w', encoding='utf-8') as fw:
                    json.dump({'posts': aggregated}, fw, ensure_ascii=False, indent=2)
            except Exception as e:
                return ActionResult(error=f"Failed to save aggregate file {aggregate_path}: {e}")

            return ActionResult(
                extracted_content=json.dumps({
                    'saved_path': aggregate_path,
                    'added_count': added,
                    'total_count': len(aggregated),
                }, ensure_ascii=False),
                include_in_memory=True
            )

        @self.registry.action(
            "Filter LinkedIn posts using LLM for topical relevance (AI, Real Estate, AI in Real Estate) and age < 6 months. "
            "First saves ALL posts to file (default ./links.json), then runs LLM filtering, applies a max post limit, and "
            "REWRITES the file keeping only relevant posts."
        )
        async def filter_posts_and_save(posts_json: str, browser: BrowserContext, page_extraction_llm: Optional[BaseChatModel] = None, max_age_months: int = 6, save_path: str = "./links.json", context: Any = None, max_keep: int = 5):
            import os, re, json
            from typing import List, Dict, Any
            # Resolve LLM client
            llm_client = page_extraction_llm
            try:
                if llm_client is None and context is not None:
                    for attr in ('llm', 'planner_llm', 'model', 'chat_model'):
                        llm_client = getattr(context, attr, None) or llm_client
            except Exception:
                pass

            def parse_age_to_months(age_text: str | None) -> float | None:
                if not age_text:
                    return None
                t = str(age_text).strip().lower()
                m = re.search(r"(\d+)\s*(year|years|y)\b", t)
                if m:
                    return int(m.group(1)) * 12.0
                m = re.search(r"(\d+)\s*(month|months|mo)\b", t)
                if m:
                    return float(m.group(1))
                m = re.search(r"(\d+)\s*(week|weeks|w)\b", t)
                if m:
                    return int(m.group(1)) * (7.0 / 30.437)
                m = re.search(r"(\d+)\s*(day|days|d)\b", t)
                if m:
                    return int(m.group(1)) / 30.437
                m = re.search(r"(\d+)\s*(hour|hours|h)\b", t)
                if m:
                    return int(m.group(1)) / (24.0 * 30.437)
                m = re.search(r"(\d+)\s*(minute|minutes|min|m)\b", t)
                if m:
                    return int(m.group(1)) / (24.0 * 30.437 * 60.0)
                m = re.search(r"^(\d+)(y|mo|w|d|h|m)$", t)
                if m:
                    val, unit = int(m.group(1)), m.group(2)
                    return {
                        'y': val * 12.0,
                        'mo': float(val),
                        'w': val * (7.0 / 30.437),
                        'd': val / 30.437,
                        'h': val / (24.0 * 30.437),
                        'm': val / (24.0 * 30.437 * 60.0),
                    }[unit]
                return None

            def simple_topic_heuristic(text: str | None) -> Dict[str, Any]:
                if not text:
                    return {"related": False, "matched_topics": []}
                t = text.lower()
                topics = []
                if ("ai" in t or "artificial intelligence" in t or "genai" in t or "gpt" in t or "llm" in t):
                    topics.append("AI")
                if ("real estate" in t or "property" in t or "housing" in t or "mortgage" in t or "realtor" in t or "estate" in t):
                    topics.append("Real Estate")
                if ("ai in real estate" in t) or ("real estate" in t and ("ai" in t or "artificial intelligence" in t)):
                    topics.append("AI in Real Estate")
                topics = list(dict.fromkeys(topics))
                return {"related": len(topics) > 0, "matched_topics": topics}

            async def llm_judgement(text: str) -> Dict[str, Any]:
                if llm_client is None:
                    return simple_topic_heuristic(text)
                guidance = (
                    "Return only JSON. Determine if the LinkedIn post content is related to any of: AI, Real Estate, AI in Real Estate. "
                    "Keys: related (boolean), matched_topics (array from: 'AI','Real Estate','AI in Real Estate'). If unsure, related=false.\n\n"
                    f"Content:\n{text}\n\nJSON:"
                )
                try:
                    if hasattr(llm_client, 'ainvoke'):
                        msg = await llm_client.ainvoke(guidance)  # type: ignore
                    elif hasattr(llm_client, 'predict'):
                        out = llm_client.predict(guidance)  # type: ignore
                        class Obj: content = out
                        msg = Obj()
                    else:
                        return simple_topic_heuristic(text)
                    content = getattr(msg, 'content', None) or str(msg)
                    try:
                        import json as _json
                        return _json.loads(content)
                    except Exception:
                        try:
                            import json_repair
                            return json.loads(json_repair.repair_json(content))
                        except Exception:
                            return simple_topic_heuristic(text)
                except Exception:
                    return simple_topic_heuristic(text)

            # Parse input
            try:
                parsed = json.loads(posts_json)
            except Exception:
                try:
                    import json_repair
                    parsed = json.loads(json_repair.repair_json(posts_json))
                except Exception as e:
                    return ActionResult(error=f"Invalid posts_json: {e}")
            if isinstance(parsed, dict) and 'posts' in parsed and isinstance(parsed['posts'], list):
                posts = parsed['posts']
            elif isinstance(parsed, list):
                posts = parsed
            else:
                return ActionResult(error='posts_json must be an array or an object with key "posts"')

            # Save ALL posts first as requested
            try:
                dirn = os.path.dirname(save_path)
                if dirn:
                    os.makedirs(dirn, exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as fw:
                    json.dump({'posts': posts}, fw, ensure_ascii=False, indent=2)
            except Exception as e:
                return ActionResult(error=f"Failed to write unfiltered posts to {save_path}: {e}")

            # Reload from file to emulate filtering "from the file"
            try:
                with open(save_path, 'r', encoding='utf-8') as fr:
                    file_json = json.load(fr)
                    posts = file_json['posts'] if isinstance(file_json, dict) else file_json
            except Exception:
                pass

            # Try batch LLM filtering
            async def llm_filter_batch(items: List[Dict[str, Any]]) -> List[Dict[str, Any]] | None:
                if llm_client is None:
                    return None
                normalized = [{
                    'link': (it or {}).get('link'),
                    'content': (it or {}).get('text') or (it or {}).get('content'),
                    'age': (it or {}).get('age')
                } for it in items]
                guidance = (
                    "You will receive a JSON array of post objects: link (string), content (string or null), age (string).\n"
                    "Filter and keep only posts where: (1) content relates to any of [AI, Real Estate, AI in Real Estate], and (2) age is strictly younger than 6 months.\n"
                    "Age formats: 30m, 3h, 2d, 3w, 5mo, 1y, or phrases like '3 months ago'. Treat: 1y=12mo; 1mo=1 month; 1w=7 days; 1d=1 day; 1h=1 hour; 30m=30 minutes. Treat 6mo or older (>= 6 months) as NOT acceptable.\n"
                    "Return strictly JSON ONLY: {\"posts\":[{\"link\":str,\"content\":str|null,\"age\":str,\"matched_topics\":[\"AI\",\"Real Estate\",\"AI in Real Estate\"]}...]}.\n"
                )
                try:
                    payload = json.dumps({'posts': normalized}, ensure_ascii=False)
                    prompt = guidance + "\nPosts JSON:\n" + payload + "\n\nFiltered JSON:"
                    if hasattr(llm_client, 'ainvoke'):
                        msg = await llm_client.ainvoke(prompt)  # type: ignore
                    elif hasattr(llm_client, 'predict'):
                        out = llm_client.predict(prompt)  # type: ignore
                        class Obj: content = out
                        msg = Obj()
                    else:
                        return None
                    content = getattr(msg, 'content', None) or str(msg)
                    try:
                        obj = json.loads(content)
                    except Exception:
                        try:
                            import json_repair
                            obj = json.loads(json_repair.repair_json(content))
                        except Exception:
                            return None
                    if isinstance(obj, dict) and isinstance(obj.get('posts'), list):
                        return obj['posts']
                    if isinstance(obj, list):
                        return obj
                    return None
                except Exception:
                    return None

            filtered = await llm_filter_batch(posts)
            if filtered is None:
                filtered = []
                # Per-post judgement fallback
                for post in posts:
                    link = (post or {}).get('link')
                    text = (post or {}).get('text') or (post or {}).get('content')
                    age = (post or {}).get('age')
                    age_months = parse_age_to_months(age)
                    is_young = (age_months is not None) and (age_months < float(max_age_months))
                    judgement = await llm_judgement(text or "")
                    related = bool(judgement.get('related'))
                    topics = judgement.get('matched_topics') or []
                    if related and is_young and link:
                        filtered.append({'link': link, 'content': text, 'age': age, 'matched_topics': topics})
            else:
                # Enforce age constraint programmatically as a safety check
                coerced = []
                for p in filtered:
                    link = (p or {}).get('link')
                    content = (p or {}).get('content') or (p or {}).get('text')
                    age = (p or {}).get('age')
                    age_months = parse_age_to_months(age)
                    if link and age_months is not None and age_months < float(max_age_months):
                        topics = (p or {}).get('matched_topics') or []
                        coerced.append({'link': link, 'content': content, 'age': age, 'matched_topics': topics})
                filtered = coerced

            # Apply post limit
            try:
                mk = int(max_keep)
            except Exception:
                mk = 5
            if mk > 0:
                filtered = filtered[:mk]

            # Rewrite file to contain only filtered posts
            try:
                dirn = os.path.dirname(save_path)
                if dirn:
                    os.makedirs(dirn, exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as fw:
                    json.dump(filtered, fw, ensure_ascii=False, indent=2)
            except Exception as e:
                return ActionResult(error=f"Failed to save filtered posts to {save_path}: {e}")

            result_obj = {'saved_path': save_path, 'filtered_count': len(filtered), 'posts': filtered}
            return ActionResult(extracted_content=json.dumps(result_obj, ensure_ascii=False), include_in_memory=True)

        @self.registry.action(
            "Finalize: read aggregate file and filter via the LLM, writing relevant posts to ./links.json."
        )
        async def finalize_and_filter_posts(
            browser: BrowserContext,
            aggregate_path: str = "./links_raw.json",
            save_path: str = "./links.json",
            max_age_months: int = 6,
            page_extraction_llm: Optional[BaseChatModel] = None,
            context: Any = None,
            max_keep: int = 5,
        ):
            import os, json
            if not os.path.exists(aggregate_path):
                return ActionResult(error=f"Aggregate file not found: {aggregate_path}")
            try:
                with open(aggregate_path, 'r', encoding='utf-8') as fr:
                    data = json.load(fr)
                if isinstance(data, dict) and isinstance(data.get('posts'), list):
                    posts = data['posts']
                elif isinstance(data, list):
                    posts = data
                else:
                    return ActionResult(error='Invalid aggregate file format; expected {"posts": [...]} or an array')
            except Exception as e:
                return ActionResult(error=f"Failed to read aggregate file: {e}")

            try:
                posts_json = json.dumps({'posts': posts}, ensure_ascii=False)
            except Exception as e:
                return ActionResult(error=f"Failed to serialize posts for filtering: {e}")

            # Reuse the filter action
            return await filter_posts_and_save(
                posts_json=posts_json,
                browser=browser,
                page_extraction_llm=page_extraction_llm,
                max_age_months=max_age_months,
                save_path=save_path,
                context=context,
                max_keep=max_keep,
            )

        @self.registry.action(
            "Orchestrate complete LinkedIn collection: iteratively extract+save batches while scrolling, then finalize and filter."
        )
        async def orchestrate_linkedin_collection(
            browser: BrowserContext,
            aggregate_path: str = "./links_raw.json",
            save_path: str = "./links.json",
            per_batch_max_posts: int = 10,
            max_batches: int = 50,
            scroll_px: int = 600,
            wait_after_scroll_seconds: float = 2.0,
            consecutive_no_new_to_stop: int = 3,
            max_age_months: int = 6,
            page_extraction_llm: Optional[BaseChatModel] = None,
            context: Any = None,
            max_keep: int = 5,
            profile_url: Optional[str] = None,
            force_recent_activity: bool = True,
        ):
            import json
            stop_reason = None
            total_added = 0
            batches_run = 0
            no_new_streak = 0

            # Try to get Playwright page to scroll
            page = None
            try:
                try:
                    from playwright.async_api import Page, BrowserContext as PWBrowserContext  # type: ignore
                except Exception:
                    Page = None
                    PWBrowserContext = None
                candidate_attrs = ['page', '_page', 'current_page', 'playwright_page', 'context', '_context', 'playwright_context', 'pw_context']
                for attr in candidate_attrs:
                    obj = getattr(browser, attr, None)
                    if obj is None:
                        continue
                    if Page is not None:
                        try:
                            from playwright.async_api import Page as _P
                            if isinstance(obj, _P):
                                page = obj
                                break
                        except Exception:
                            pass
                    if PWBrowserContext is not None:
                        try:
                            from playwright.async_api import BrowserContext as _C
                            if isinstance(obj, _C):
                                pages = obj.pages
                                if pages:
                                    page = pages[-1]
                                    break
                        except Exception:
                            pass
                    if hasattr(obj, 'pages'):
                        try:
                            pages = obj.pages
                            if pages:
                                page = pages[-1]
                                break
                        except Exception:
                            pass
                if page is None and hasattr(browser, 'get_current_page'):
                    try:
                        page = await browser.get_current_page()  # type: ignore
                    except Exception:
                        pass
            except Exception:
                page = None

            # Optional pre-navigation hack: go directly to /recent-activity
            if page is not None:
                try:
                    current_url = page.url or ""
                except Exception:
                    current_url = ""
                try:
                    target = None
                    if profile_url:
                        u = str(profile_url).strip()
                        if u.endswith('/'):
                            u = u[:-1]
                        if 'recent-activity' not in u:
                            u = u + '/recent-activity'
                        target = u
                    elif force_recent_activity and current_url and 'linkedin.com/in/' in current_url and 'recent-activity' not in current_url:
                        # Derive from current profile URL
                        base = current_url.split('?', 1)[0].split('#', 1)[0]
                        if base.endswith('/'):
                            base = base[:-1]
                        if 'recent-activity' not in base:
                            target = base + '/recent-activity'
                    if target:
                        try:
                            await page.goto(target)
                        except Exception:
                            # Fallback to open in same tab via location
                            try:
                                await page.evaluate("url => window.location.href = url", target)
                            except Exception:
                                pass
                except Exception:
                    pass

            # Iterate batches
            for i in range(max(1, int(max_batches))):
                batches_run += 1
                res = await linkedin_extract_and_save(
                    browser=browser,
                    max_posts=per_batch_max_posts,
                    aggregate_path=aggregate_path,
                )
                added = 0
                try:
                    if res and res.extracted_content:
                        obj = json.loads(res.extracted_content)
                        added = int(obj.get('added_count', 0))
                except Exception:
                    added = 0
                total_added += added
                if added == 0:
                    no_new_streak += 1
                else:
                    no_new_streak = 0

                # Stop if no new posts repeatedly
                if no_new_streak >= max(1, int(consecutive_no_new_to_stop)):
                    stop_reason = f"no_new_for_{no_new_streak}_batches"
                    break

                # Scroll for next batch if possible
                if page is not None:
                    try:
                        await page.evaluate("window.scrollBy(0, arguments[0]);", scroll_px)
                    except Exception:
                        # If scroll fails, try ArrowDown fallback
                        try:
                            await page.keyboard.press('PageDown')
                        except Exception:
                            pass
                # Wait for content load
                try:
                    import asyncio as _asyncio
                    await _asyncio.sleep(float(wait_after_scroll_seconds))
                except Exception:
                    pass

            if stop_reason is None:
                stop_reason = "max_batches_reached"

            # Finalize and filter
            filt = await finalize_and_filter_posts(
                browser=browser,
                aggregate_path=aggregate_path,
                save_path=save_path,
                max_age_months=max_age_months,
                page_extraction_llm=page_extraction_llm,
                context=context,
                max_keep=max_keep,
            )

            filt_obj = {}
            try:
                filt_obj = json.loads(filt.extracted_content) if filt and filt.extracted_content else {}
            except Exception:
                filt_obj = {}

            # Optional: quick summary
            try:
                summ = await summarize_links_status(browser=browser, aggregate_path=aggregate_path, filtered_path=save_path)
                summ_obj = json.loads(summ.extracted_content) if summ and summ.extracted_content else {}
            except Exception:
                summ_obj = {}

            result = {
                'aggregate_path': aggregate_path,
                'save_path': save_path,
                'batches_run': batches_run,
                'total_added': total_added,
                'stop_reason': stop_reason,
                'filtered_count': filt_obj.get('filtered_count'),
                'summary': summ_obj,
            }
            return ActionResult(extracted_content=json.dumps(result, ensure_ascii=False), include_in_memory=True)

        @self.registry.action(
            "Summarize status of aggregate and filtered link files. Reports totals, unique links, and simple diffs."
        )
        async def summarize_links_status(
            browser: BrowserContext,
            aggregate_path: str = "./links_raw.json",
            filtered_path: str = "./links.json",
        ):
            import os, json

            def _load_posts(path: str) -> list:
                if not os.path.exists(path):
                    return []
                try:
                    with open(path, 'r', encoding='utf-8') as fr:
                        data = json.load(fr)
                    if isinstance(data, dict) and isinstance(data.get('posts'), list):
                        return data['posts']
                    if isinstance(data, list):
                        return data
                except Exception:
                    return []
                return []

            agg_posts = _load_posts(aggregate_path)
            fil_posts = _load_posts(filtered_path)

            agg_links = { (p or {}).get('link') for p in agg_posts if (p or {}).get('link') }
            fil_links = { (p or {}).get('link') for p in fil_posts if (p or {}).get('link') }

            summary = {
                'aggregate': {
                    'path': aggregate_path,
                    'exists': os.path.exists(aggregate_path),
                    'total': len(agg_posts),
                    'unique_links': len(agg_links),
                },
                'filtered': {
                    'path': filtered_path,
                    'exists': os.path.exists(filtered_path),
                    'total': len(fil_posts),
                    'unique_links': len(fil_links),
                },
                'diff': {
                    'unique_not_filtered': len(agg_links - fil_links),
                    'unique_extra_in_filtered': len(fil_links - agg_links),
                }
            }

            return ActionResult(
                extracted_content=json.dumps(summary, ensure_ascii=False),
                include_in_memory=True,
            )

    @time_execution_sync('--act')
    async def act(
            self,
            action: ActionModel,
            browser_context: Optional[BrowserContext] = None,
            #
            page_extraction_llm: Optional[BaseChatModel] = None,
            sensitive_data: Optional[Dict[str, str]] = None,
            available_file_paths: Optional[list[str]] = None,
            #
            context: Context | None = None,
    ) -> ActionResult:
        """Execute an action"""

        try:
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    if action_name.startswith("mcp"):
                        # this is a mcp tool
                        logger.debug(f"Invoke MCP tool: {action_name}")
                        mcp_tool = self.registry.registry.actions.get(action_name).function
                        result = await mcp_tool.ainvoke(params)
                    else:
                        result = await self.registry.execute_action(
                            action_name,
                            params,
                            browser=browser_context,
                            page_extraction_llm=page_extraction_llm,
                            sensitive_data=sensitive_data,
                            available_file_paths=available_file_paths,
                            context=context,
                        )

                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            raise e

    async def setup_mcp_client(self, mcp_server_config: Optional[Dict[str, Any]] = None):
        self.mcp_server_config = mcp_server_config
        if self.mcp_server_config:
            self.mcp_client = await setup_mcp_client_and_tools(self.mcp_server_config)
            self.register_mcp_tools()

    def register_mcp_tools(self):
        """
        Register the MCP tools used by this controller.
        """
        if self.mcp_client:
            for server_name in self.mcp_client.server_name_to_tools:
                for tool in self.mcp_client.server_name_to_tools[server_name]:
                    tool_name = f"mcp.{server_name}.{tool.name}"
                    self.registry.registry.actions[tool_name] = RegisteredAction(
                        name=tool_name,
                        description=tool.description,
                        function=tool,
                        param_model=create_tool_param_model(tool),
                    )
                    logger.info(f"Add mcp tool: {tool_name}")
                logger.debug(
                    f"Registered {len(self.mcp_client.server_name_to_tools[server_name])} mcp tools for {server_name}")
        else:
            logger.warning(f"MCP client not started.")

    async def close_mcp_client(self):
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)
