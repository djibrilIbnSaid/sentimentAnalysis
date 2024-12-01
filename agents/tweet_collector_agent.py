import os
import asyncio
import pandas as pd
import json
from langchain_core.messages import HumanMessage
from twscrape import API, gather
from twscrape.logger import set_log_level
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

username = os.getenv("TWITTER_USERNAME")
account_password = os.getenv("TWITTER_ACCOUNT_PASSWORD")
email = os.getenv("TWITTER_EMAIL")
password = os.getenv("TWITTER_EMAIL_PASSWORD")

class TweetCollectorAgent:
    def __init__(self, mode='term', number=100, output_file='data/tweets.json'):
        self.name = 'TweetCollectorAgent'
        self.mode = mode
        self.number = number
        self.query = None
        self.output_file = output_file
        self.api = API()

    async def setup_accounts(self):
        """
        Set up Twitter accounts and log in to the API.
        """
        try:
            # Add accounts and log in
            # await self.api.pool.delete_accounts(username)
            await self.api.pool.add_account(username, account_password, email, password)
            await self.api.pool.login_all()
            print(f"Accounts setup successfully")
        except Exception as e:
            print(f"Error in account setup: {e}")
            print("The DataCleaner agent is going to use the existing twiter's data stored locally ")
            raise

    async def fetch_tweets(self):
        """
        Fetch tweets based on the query and number.

        Returns:
            List: List of tweets fetched
        """
        try:
            # Fetch tweets based on the query and number
            tweets = await gather(self.api.search(self.query+" lang:fr", limit=self.number))
            print(f"Successfully fetched {len(tweets)} tweets")
            return tweets
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            raise

    async def save_to_json(self, tweets):
        """
        Save tweet details to a JSON file.

        Args:
            tweets: List of tweets to save to JSON
        """
        try:
            tweets_list = []
            for tweet in tweets:
                # Extract tweet details
                tweet_data = {
                    "tweet_id": tweet.id,
                    "tweet_url": tweet.url,
                    "tweet_date": tweet.date.strftime("%Y-%m-%d %H:%M:%S"),  # Convert datetime to string
                    "tweet_language": tweet.lang,
                    "tweet_content": tweet.rawContent,
                    "reply_count": tweet.replyCount,
                    "retweet_count": tweet.retweetCount,
                    "like_count": tweet.likeCount,
                    "quote_count": tweet.quoteCount,
                    "view_count": tweet.viewCount,
                }

                # Extract user details from the tweet
                user_data = tweet.user
                tweet_data.update({
                    "user_id": user_data.id,
                    "user_username": user_data.username,
                    "user_display_name": user_data.displayname,
                    "user_profile_url": user_data.url,
                    "user_description": user_data.rawDescription,
                    "user_created_at": user_data.created.strftime("%Y-%m-%d %H:%M:%S"),  # Convert datetime to string
                    "user_followers_count": user_data.followersCount,
                    "user_friends_count": user_data.friendsCount,
                    "user_statuses_count": user_data.statusesCount,
                    "user_favourites_count": user_data.favouritesCount,
                    "user_media_count": user_data.mediaCount,
                    "user_location": user_data.location,
                    "user_verified": user_data.verified,
                    "user_blue_check_type": user_data.blueType,
                })

                # Optionally, check if the tweet is a quoted tweet and retrieve those details
                if tweet.quotedTweet:
                    quoted_tweet = tweet.quotedTweet
                    tweet_data["quoted_tweet_id"] = quoted_tweet.id
                    tweet_data["quoted_tweet_url"] = quoted_tweet.url
                    tweet_data["quoted_tweet_date"] = quoted_tweet.date.strftime("%Y-%m-%d %H:%M:%S")  # Convert datetime to string
                    tweet_data["quoted_tweet_content"] = quoted_tweet.rawContent
                    tweet_data["quoted_tweet_like_count"] = quoted_tweet.likeCount
                    tweet_data["quoted_tweet_retweet_count"] = quoted_tweet.retweetCount
                    tweet_data["quoted_tweet_language"] = quoted_tweet.lang

                # Append tweet data to list
                tweets_list.append(tweet_data)

            # Write tweet details to a JSON file
            with open(self.output_file, "w", encoding="utf-8") as json_file:
                json.dump(tweets_list, json_file, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving to JSON: {e}")
            raise

    async def crawl_tweets(self):
        """
        Crawl tweets based on the query and number.
        """
        try:
            # Set up logging, accounts, and fetch/save tweets
            set_log_level("DEBUG")
            await self.setup_accounts()
            tweets = await self.fetch_tweets()
            await self.save_to_json(tweets)
        except Exception as e:
            print(f"Error during crawling: {e}")
            raise

    def invoke(self, state):
        """
        Méthode principale pour l'agent

        Args:
            state: l'état actuel de l'agent

        Returns:
            dict: l'état mis à jour de l'agent
        """
        
        try:
            # Get query from the state
            self.query = state.get("context", "")  # Query can be passed via state

            if not self.query:
                raise ValueError("Query not provided in state.")

            # Check if an event loop is already running
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop:
                # Event loop is already running
                task = loop.create_task(self.crawl_tweets())
                loop.run_until_complete(task)
            else:
                # No running event loop, safe to use asyncio.run()
                asyncio.run(self.crawl_tweets())

            # Return the state with the new message and data path
            return {
                "messages": state["messages"] + [HumanMessage(content=f"Action effectuée par l'agent {self.name}")],
                "data": self.output_file,
                "context": state.get("context", {})
            }

        except Exception as e:
            print(f"Error in invoking agent: {e}")
            return {
                "messages": state["messages"] + [HumanMessage(content=f"Erreur: {e}")],
                "data": self.output_file,
                "context": state.get("context", {})
            }
