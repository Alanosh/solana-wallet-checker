import random
import time
import asyncio
import aiohttp
import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event
from solana.keypair import Keypair
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from solana.transaction import Transaction, TransactionInstruction
from solana.publickey import PublicKey
from solana.system_program import TransferParams, transfer
from bip_utils import Bip39SeedGenerator, Bip39MnemonicGenerator, Bip39MnemonicValidator
from cryptography.fernet import Fernet
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('wallet_checker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Thread-safe file lock
file_lock = Lock()

# Encryption key for output file
ENCRYPTION_KEY_FILE = 'encryption_key.key'
fernet = None

def setup_encryption():
    """Set up encryption for output file."""
    global fernet
    if os.path.exists(ENCRYPTION_KEY_FILE):
        with open(ENCRYPTION_KEY_FILE, 'rb') as key_file:
            key = key_file.read()
    else:
        key = Fernet.generate_key()
        with open(ENCRYPTION_KEY_FILE, 'wb') as key_file:
            key_file.write(key)
    fernet = Fernet(key)

class ApiService:
    """Handles interactions with the Solana blockchain."""
    def __init__(self, rpc_url):
        """
        Initialize the API service.

        Args:
            rpc_url (str): Solana RPC endpoint URL.
        """
        self.client = Client(rpc_url)
        self.rpc_url = rpc_url

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def get_balance(self, public_key):
        """
        Fetch the balance of a Solana public key.

        Args:
            public_key (PublicKey): The public key to check.

        Returns:
            float: Balance in SOL, or None if an error occurs.
        """
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getBalance",
                    "params": [str(public_key)]
                }
                async with session.post(self.rpc_url, json=payload) as resp:
                    resp.raise_for_status()
                    response = await resp.json()
                    balance = response.get("result", {}).get("value", 0) / 1e9
                    return balance
        except (aiohttp.ClientError, ValueError) as e:
            logger.error(f"Error fetching balance for {public_key}: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def get_transaction_history(self, public_key, inactivity_days):
        """
        Check if the wallet has transactions within the last inactivity_days.

        Args:
            public_key (PublicKey): The public key to check.
            inactivity_days (int): Number of days to check for inactivity.

        Returns:
            bool: True if no transactions in the last inactivity_days, False otherwise.
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=inactivity_days)
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": [
                        str(public_key),
                        {"limit": 1000}  # Adjust limit if needed
                    ]
                }
                async with session.post(self.rpc_url, json=payload) as resp:
                    resp.raise_for_status()
                    response = await resp.json()
                    signatures = response.get("result", [])

                    if not signatures:
                        logger.debug(f"No transactions found for {public_key}")
                        return True

                    # Check the most recent transaction's block time
                    for sig in signatures:
                        block_time = sig.get("blockTime")
                        if block_time:
                            tx_time = datetime.utcfromtimestamp(block_time)
                            if tx_time >= cutoff_time:
                                logger.debug(f"Found recent transaction for {public_key} at {tx_time}")
                                return False
                    logger.debug(f"No transactions in last {inactivity_days} days for {public_key}")
                    return True
        except (aiohttp.ClientError, ValueError) as e:
            logger.error(f"Error fetching transaction history for {public_key}: {e}")
            return False  # Assume active if error occurs to avoid false positives

    def send_transaction(self, transaction, keypair):
        """
        Send a transaction to the Solana blockchain (unused).

        Args:
            transaction (Transaction): The transaction to send.
            keypair (Keypair): The keypair to sign the transaction.

        Returns:
            str: Transaction ID, or None if an error occurs.
        """
        try:
            response = self.client.send_transaction(transaction, keypair, opts=TxOpts(skip_confirmation=False))
            return response.get("result")
        except Exception as e:
            logger.error(f"Error sending transaction: {e}")
            return None

class WalletManager:
    """Manages Solana wallet operations."""
    def __init__(self, api_service):
        """
        Initialize the wallet manager.

        Args:
            api_service (ApiService): The API service instance.
        """
        self.api_service = api_service

    def create_wallet(self, mnemonic):
        """
        Create a Solana wallet from a mnemonic phrase.

        Args:
            mnemonic (str): The BIP-39 mnemonic phrase.

        Returns:
            Keypair: The generated Solana keypair.
        """
        try:
            seed = Bip39SeedGenerator(mnemonic).Generate()
            keypair = Keypair.from_seed(seed[:32])
            return keypair
        except Exception as e:
            logger.error(f"Error creating wallet from mnemonic: {e}")
            return None

    async def fetch_balance(self, public_key):
        """
        Fetch the balance of a wallet.

        Args:
            public_key (PublicKey): The public key to check.

        Returns:
            float: Balance in SOL, or None if an error occurs.
        """
        return await self.api_service.get_balance(public_key)

def generate_mnemonic():
    """
    Generate a cryptographically secure BIP-39 mnemonic phrase.

    Returns:
        str: A 12-word mnemonic phrase.
    """
    try:
        return Bip39MnemonicGenerator().FromWordsNumber(12)
    except Exception as e:
        logger.error(f"Error generating mnemonic: {e}")
        return None

def load_word_list(file_path):
    """
    Load a BIP-39 word list from a file and validate it.

    Args:
        file_path (str): Path to the word list file.

    Returns:
        list: List of words, or empty list if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            word_list = [line.strip() for line in file.readlines()]
        if len(word_list) < 2048:
            logger.warning(f"Word list has {len(word_list)} words; expected ~2048 for BIP-39.")
        return word_list
    except (FileNotFoundError, UnicodeDecodeError) as e:
        logger.error(f"Error loading word list from {file_path}: {e}")
        return []

def save_wallet_to_file(mnemonic, balance):
    """
    Save the mnemonic and balance to an encrypted file.

    Args:
        mnemonic (str): The mnemonic phrase.
        balance (float): The wallet balance in SOL.
    """
    try:
        data = f"Recovery Phrase: {mnemonic}\nBalance: {balance:.6f} SOL\n\n"
        encrypted_data = fernet.encrypt(data.encode())
        with file_lock:
            with open('found_wallets.enc', 'ab') as file:
                file.write(encrypted_data + b'\n')
        logger.info(f"Saved wallet with balance {balance:.6f} SOL to encrypted file.")
    except Exception as e:
        logger.error(f"Error saving wallet to file: {e}")

async def check_wallet(api_service, wallet_manager, stop_event, balance_threshold, inactivity_days, wallet_counter):
    """
    Check a wallet for a non-zero balance and transaction inactivity.

    Args:
        api_service (ApiService): The API service instance.
        wallet_manager (WalletManager): The wallet manager instance.
        stop_event (Event): Event to signal when to stop.
        balance_threshold (float): Balance threshold to check.
        inactivity_days (int): Days to check for transaction inactivity.
        wallet_counter (dict): Shared counter for tracking wallets checked.

    Returns:
        bool: True if a wallet meets balance and inactivity criteria, False otherwise.
    """
    if stop_event.is_set():
        return False

    mnemonic = generate_mnemonic()
    if not mnemonic:
        return False

    if not Bip39MnemonicValidator().IsValid(mnemonic):
        logger.debug(f"Generated invalid mnemonic: {mnemonic}")
        return False

    sender_keypair = wallet_manager.create_wallet(mnemonic)
    if not sender_keypair:
        return False

    balance = await wallet_manager.fetch_balance(sender_keypair.public_key)
    wallet_counter['checked'] += 1

    if balance is None or balance <= 0:
        return False

    save_wallet_to_file(mnemonic, balance)

    if balance >= balance_threshold:
        is_inactive = await api_service.get_transaction_history(sender_keypair.public_key, inactivity_days)
        if is_inactive:
            logger.info(f"Found inactive wallet with balance {balance:.6f} SOL (no tx in {inactivity_days} days)")
            logger.info(f"Recovery phrase: {mnemonic}")
            stop_event.set()
            return True
        else:
            logger.info(f"Wallet with balance {balance:.6f} SOL has recent transactions")

    logger.info(f"Found wallet with balance {balance:.6f} SOL (below threshold {balance_threshold})")
    return False

async def main(args):
    """Main function to run the wallet checker."""
    logger.info("Starting Solana wallet checker.")
    logger tó) warning(
        "WARNING: This script generates random mnemonic phrases to check for funded Solana wallets. "
        "Using it to access wallets you do not own is ILLEGAL and UNETHICAL. "
        "Ensure you have permission to access any wallets checked."
    )
    confirm = input("Type 'I UNDERSTAND' to continue: ")
    if confirm != "I UNDERSTAND":
        logger.error("User did not confirm. Exiting.")
        return

    setup_encryption()
    api_service = ApiService(args.rpc_url)
    wallet_manager = WalletManager(api_service)

    word_list = load_word_list(args.word_list)
    if not word_list:
        logger.error("No valid word list provided. Exiting.")
        return

    stop_event = Event()
    wallet_counter = {'checked': 0, 'found': 0}
    tasks = []

    async def check_and_log():
        nonlocal wallet_counter
        result = await check_wallet(api_service, wallet_manager, stop_event, args.balance_threshold, args.inactivity_days, wallet_counter)
        if result:
            wallet_counter['found'] += 1

    for _ in range(args.num_checks):
        if stop_event.is_set():
            break
        tasks.append(check_and_log())

    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info(f"Checked {wallet_counter['checked']} wallets.")
    logger.info(f"Found {wallet_counter['found']} wallets with balance >= {args.balance_threshold} SOL and no transactions in {args.inactivity_days} days.")
    if os.path.exists('found_wallets.enc'):
        logger.info("Non-zero balance wallets saved to 'found_wallets.enc' (encrypted).")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Solana wallet balance and inactivity checker.")
    parser.add_argument(
        '--rpc-url',
        default='https://api.mainnet-beta.solana.com',
        help='Solana RPC endpoint URL'
    )
    parser.add_argument(
        '--word-list',
        default='word_list.txt',
        help='Path to BIP-39 word list file'
    )
    parser.add_argument(
        '--balance-threshold',
        type=float,
        default=0.923,
        help='Balance threshold to stop execution (in SOL)'
    )
    parser.add_argument(
        '--inactivity-days',
        type=int,
        default=30,
        help='Number of days to check for transaction inactivity'
    )
    parser.add_argument(
        '--num-checks',
        type=int,
        default=100,
        help='Number of wallets to check'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=min(os.cpu_count() * 2, 50),
        help='Maximum number of concurrent workers'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
